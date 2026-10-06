"""pigemma.h2.d1_lowrank_residual - D1 dense-residual feasibility (brief 2026-10-05, section 6).

APPROXIMATE-computation research track, desktop only, separate from the exact MoE caching work.

Reference: the 31B Google QAT Q4_0 GGUF dequantised to float32 (``W_ref``). The original BF16
checkpoint is NOT used, so the quantisation error of Q4_0 itself is not measured here; what is
measured instead is llama.cpp's runtime numerical error against the float32 reference.

Representation (pre-registered in results/dense/preregistration.json before any evaluation):

* base: activation-aware low-rank ``W_base = U_r U_r^T W`` with ``U_r`` the top-r left singular
  vectors of ``W L`` where ``L L^T = C + lam*mean(diag C)*I`` and ``C = E[x x^T]`` over the
  CALIBRATION activations ('whiten', primary; it minimises ``||(W - W_base) X||_F`` on the
  calibration set for rank r) or ``C -> diag(C)`` ('diag', ASVD-like, secondary). Factors
  ``A = U_r`` [n_out, r] and ``B = U_r^T W`` [r, n_in] are stored 8-bit absmax block-32 + f16
  scale (1.0625 B/weight) and the evaluation uses those quantised factors.
* residual: ``R = W_ref - A_q B_q`` in blocks of 128 input columns (primary) or 128 output rows.
  A selected block is applied exactly, with the reference Q4_0 bytes of that block as payload
  (the Q4_0 blocks of 32 inputs tile a 128-column block, so the re-layout is byte-exact), i.e.
  ``y = A_q (B_q x)|unselected + W_ref[:, sel] x_sel``; base + all blocks == W_ref.

The oracle selections use the true residual contributions (not deployable): column blocks are
read greedily by ``||R_g x_g||``; row blocks are optimal (disjoint output coordinates). The cheap
online column selector is the norm bound ``b_g = ||R_g||_2 ||x_g||``.

CLI::

    python -m pigemma.h2.d1_lowrank_residual --cal DIR --heldout DIR --layers 2,30,57 \
        --out results/dense [--threads 6] [--ranks 128,256,512,1024,2048] [--composed-ranks 256,512,1024]

Writes ``<out>/L<il>.json`` per layer and prints a short table. Labels: every number it writes is
MEASURED (computed on this host from the captured activations and the real weights) unless the
key says ``projected``.
"""

from __future__ import annotations

import os

_T = os.environ.get("PIGEMMA_THREADS", "6")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _T)

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pigemma.h2 import decompose as dc

MAP31 = "traces/h2/map31.json"
GC = 128                       # block width (columns or rows)
B_Q4_0 = 18 / 32               # bytes per weight
B_Q3_K = 110 / 256
B_Q8 = 34 / 32                 # 8-bit absmax block-32 + f16 scale
B_Q3 = 0.4375
B_Q2 = 0.3125
EPS_MAT = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
FFN_TARGETS = (1e-2, 5e-3, 1e-3)
FIX_F = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
LAM = 0.01


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def stats(a: Any) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "max": float(a.max())}


# ------------------------------------------------------------------------------------------
# Activations
# ------------------------------------------------------------------------------------------
def load_acts(d: Path, layer: int) -> dict[str, Any]:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    T = int(meta["n_tokens"])
    out: dict[str, Any] = {"meta": meta, "T": T}
    for k in ("ffn_norm", "ffn_geglu", "ffn_out"):
        a = np.load(d / f"L{layer}.{k}.npy", mmap_mode="r")[:T]
        out[k] = np.array(a, dtype=np.float32)
    ph = np.fromfile(d / "phase.u8", dtype=np.uint8)[:T] if (d / "phase.u8").is_file() else np.zeros(T, np.uint8)
    fin = np.isfinite(out["ffn_norm"]).all(1) & np.isfinite(out["ffn_geglu"]).all(1) & np.isfinite(out["ffn_out"]).all(1)
    for k in ("ffn_norm", "ffn_geglu", "ffn_out"):
        out[k] = out[k][fin]
    out["phase"] = ph[fin]
    out["n_dropped_nonfinite"] = int((~fin).sum())
    return out


# ------------------------------------------------------------------------------------------
# Q3_K (numpy port of llama.cpp ggml-quants.c quantize_row_q3_K_impl, imatrix path)
# ------------------------------------------------------------------------------------------
def make_qx_quants_vec(x: np.ndarray, nmax: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised make_qx_quants(n, nmax, x, L, rmse_type=1, qw=w) over rows of x [N, n].

    Returns (scale [N], L [N, n] in 0..2*nmax-1). Search over iscale = -(nmax + 0.1*is)/max,
    is = 0, -9..9 (same order and acceptance test as the C code; float32 arithmetic, pairwise
    sums instead of sequential ones, so not guaranteed bit-identical to llama.cpp).
    """
    x = x.astype(np.float32, copy=False)
    w = w.astype(np.float32, copy=False)
    N = x.shape[0]
    idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(N), idx]
    zero = np.abs(mx) < 1e-15
    mx_s = np.where(zero, np.float32(1), mx).astype(np.float32)

    def cand(is_: int):
        iscale = (np.float32(-(nmax + np.float32(0.1) * is_)) / mx_s).astype(np.float32)
        l = np.rint(iscale[:, None] * x)
        np.clip(l, -nmax, nmax - 1, out=l)
        wl = w * l
        return l, (wl * x).sum(1, dtype=np.float32), (wl * l).sum(1, dtype=np.float32)

    L, slx, sl2 = cand(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sl2 > 0, slx / sl2, 0).astype(np.float32)
    best = scale * slx
    for is_ in list(range(-9, 0)) + list(range(1, 10)):
        l2, slx2, sl22 = cand(is_)
        upd = (sl22 > 0) & (slx2 * slx2 > best * sl22)
        if upd.any():
            L = np.where(upd[:, None], l2, L)
            with np.errstate(divide="ignore", invalid="ignore"):
                ns = np.where(upd, slx2 / np.where(sl22 > 0, sl22, 1), scale).astype(np.float32)
            scale = ns
            best = np.where(upd, ns * slx2, best).astype(np.float32)
    scale = np.where(zero, np.float32(0), scale)
    L = np.where(zero[:, None], -nmax, L)
    return scale, (L + nmax).astype(np.int16)


def q3k_rows(Xr: np.ndarray, imat: np.ndarray, *, want_codes: bool = False):
    """Quantise rows Xr [R, n] (n % 256 == 0) to Q3_K with importance imat [n]; return dequant (and codes)."""
    R, n = Xr.shape
    nb = n // 256
    X = Xr.reshape(R * nb, 256).astype(np.float32)
    sigma2 = 2 * (X * X).sum(1) / 256
    xs = X.reshape(-1, 16, 16)
    qw = np.broadcast_to(imat.reshape(1, nb, 256), (R, nb, 256)).reshape(-1, 16, 16).astype(np.float32)
    weight = qw * np.sqrt(sigma2[:, None, None] + xs * xs)
    sw = weight.sum(2)
    scales, _ = make_qx_quants_vec(xs.reshape(-1, 16), 4, weight.reshape(-1, 16))
    scales = scales.reshape(-1, 16)
    d_block, Ls = make_qx_quants_vec(scales, 32, sw)
    d16 = d_block.astype(np.float16).astype(np.float32)
    sc = (Ls.astype(np.int32) - 32).astype(np.float32)
    dsub = d16[:, None] * sc                                  # [Nsb, 16]
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.rint(xs / dsub[..., None])
    l = np.where(dsub[..., None] == 0, 0, l)
    np.clip(l, -4, 3, out=l)
    deq = (dsub[..., None] * l).astype(np.float32).reshape(R, n)
    if want_codes:
        return deq, d16, Ls, (l + 4).astype(np.uint8).reshape(-1, 256)
    return deq


def q3k_pack(d16: np.ndarray, Ls: np.ndarray, L: np.ndarray) -> bytes:
    """Pack super-blocks into ggml block_q3_K bytes (hmask[32], qs[64], scales[12], d f16)."""
    n = L.shape[0]
    L = L.astype(np.uint8).copy()
    hmask = np.zeros((n, 32), np.uint8)
    for j in range(256):
        m, bit = j % 32, j // 32
        hi = L[:, j] > 3
        hmask[hi, m] |= np.uint8(1 << bit)
        L[hi, j] -= 4
    qs = np.zeros((n, 64), np.uint8)
    for j in (0, 128):
        for l in range(32):
            qs[:, j // 4 + l] = L[:, j + l] | (L[:, j + l + 32] << 2) | (L[:, j + l + 64] << 4) | (L[:, j + l + 96] << 6)
    sc = np.zeros((n, 12), np.uint8)
    Ls = Ls.astype(np.int32)
    for j in range(16):
        l = Ls[:, j]
        if j < 8:
            sc[:, j] = (l & 0xF).astype(np.uint8)
        else:
            sc[:, j - 8] |= ((l & 0xF) << 4).astype(np.uint8)
        l = l >> 4
        sc[:, j % 4 + 8] |= (l << (2 * (j // 4))).astype(np.uint8)
    d = d16.astype(np.float16).view(np.uint8).reshape(n, 2)
    return np.concatenate([hmask, qs, sc, d], axis=1).tobytes()


def q3k_matrix(W: np.ndarray, imat: np.ndarray, verify_rows: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.empty_like(W)
    step = max(1, (1 << 21) // W.shape[1])
    ver: dict[str, Any] = {}
    for r0 in range(0, W.shape[0], step):
        r1 = min(W.shape[0], r0 + step)
        if r0 == 0 and verify_rows:
            deq, d16, Ls, L = q3k_rows(W[:verify_rows], imat, want_codes=True)
            raw = np.frombuffer(q3k_pack(d16, Ls, L), dtype=np.uint8)
            from gguf import quants as gq, GGMLQuantizationType as QT
            ref = gq.dequantize(raw, QT.Q3_K).reshape(verify_rows, -1)
            ver = {"rows": verify_rows, "bytes": int(raw.size),
                   "bytes_expected": verify_rows * W.shape[1] // 256 * 110,
                   "max_abs_diff_vs_gguf_py_dequant": float(np.abs(ref - deq).max()),
                   "exact_match_fraction": float((ref == deq).mean())}
        out[r0:r1] = q3k_rows(W[r0:r1], imat)
    return out, ver


# ------------------------------------------------------------------------------------------
# Base fit
# ------------------------------------------------------------------------------------------
def fit_U(W: torch.Tensor, Xc: torch.Tensor, kind: str, rmax: int) -> tuple[torch.Tensor, np.ndarray]:
    """Top-rmax left singular vectors of W L (whiten) or W diag(s) (diag); returns (U [n_out, rmax], sigma)."""
    n_out, n_in = W.shape
    T = Xc.shape[0]
    if n_in <= n_out:
        # input side: C [n_in, n_in]
        C = (Xc.T @ Xc).double() / T
        m = float(torch.diagonal(C).mean())
        if kind == "whiten":
            Creg = C + LAM * m * torch.eye(n_in, dtype=torch.float64)
            L = torch.linalg.cholesky(Creg).float()
            M = W @ L
        else:
            s = torch.sqrt(torch.diagonal(C) + 1e-6 * m).float()
            M = W * s[None, :]
        G = (M.T.double() @ M.double())
        ev, V = torch.linalg.eigh(G)
        ev, V = ev.flip(0), V.flip(1)
        sig = torch.sqrt(torch.clamp(ev, min=0))
        Vr = V[:, :rmax].float()
        U = (M @ Vr) / sig[:rmax].float()[None, :]
        # re-orthonormalise (float32 rounding)
        U, _ = torch.linalg.qr(U)
        return U.contiguous(), sig.numpy()
    # output side: K = W C W^T [n_out, n_out]
    if kind == "whiten":
        Y = Xc @ W.T                                    # [T, n_out]
        K = (Y.T.double() @ Y.double()) / T
        m = float((Xc.double() ** 2).sum() / (T * n_in))
        K += LAM * m * (W.double() @ W.T.double())
    else:
        c = (Xc.double() ** 2).sum(0) / T
        m = float(c.mean())
        Ws = W.double() * torch.sqrt(c + 1e-6 * m)[None, :]
        K = Ws @ Ws.T
    ev, U = torch.linalg.eigh(K)
    ev, U = ev.flip(0), U.flip(1)
    sig = torch.sqrt(torch.clamp(ev, min=0))
    return U[:, :rmax].float().contiguous(), sig.numpy()


def q8(A: torch.Tensor) -> torch.Tensor:
    Aq, _ = dc.group_quant(A.numpy(), 8, 32)
    return torch.from_numpy(Aq)


# ------------------------------------------------------------------------------------------
# Selection analysis
# ------------------------------------------------------------------------------------------
def greedy_order(C: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Oracle order: repeatedly add the block with the largest error reduction.

    C [tb, G, n] block contributions, r [tb, n] = sum_g C_g (the residual output). The gain of
    adding block g to the already-read set is ||e||^2 - ||e - C_g||^2 = 2<C_g, e> - ||C_g||^2
    with e the still-missing part; <C_h, e> is updated from the contribution Gram matrix.
    """
    tb, G, _ = C.shape
    Gm = torch.bmm(C, C.transpose(1, 2)).double()                     # [tb, G, G]
    pr = torch.bmm(C, r[:, :, None]).squeeze(2).double()              # [tb, G]
    dg = torch.diagonal(Gm, dim1=1, dim2=2)
    avail = torch.ones(tb, G, dtype=torch.bool)
    order = torch.empty(tb, G, dtype=torch.long)
    ar = torch.arange(tb)
    ninf = torch.tensor(float("-inf"), dtype=torch.float64)
    for step in range(G):
        gain = torch.where(avail, 2 * pr - dg, ninf)
        g = gain.argmax(dim=1)
        order[:, step] = g
        avail[ar, g] = False
        pr = pr - Gm[ar, g, :]
    return order


def col_analysis(Rg: torch.Tensor, X: torch.Tensor, r: torch.Tensor, ynorm: torch.Tensor,
                 bnd: torch.Tensor | None, need_S: bool = False, static_order: torch.Tensor | None = None):
    """Column-block oracle + online orders for a token batch.

    Rg [G, n_out, GC] residual blocks; X [tb, n_in]; r = y_ref - y_base [tb, n_out]; ynorm [tb].
    Returns err_or [tb, G+1] (rel. error after reading k blocks in the greedy oracle order),
    err_on [tb, G+1] (same in the online bound order b_g = ||R_g||_2 ||x_g||; None without bnd),
    and optionally the cumulative sums in the oracle order [tb, G, n_out].
    """
    G = Rg.shape[0]
    tb = X.shape[0]
    Xg = X.view(tb, G, GC).permute(1, 2, 0)               # [G, GC, tb]
    C = torch.bmm(Rg, Xg).permute(2, 0, 1).contiguous()   # [tb, G, n_out]
    out = []
    S_or = None
    orders = [greedy_order(C, r)]
    if bnd is not None:
        xn = X.view(tb, G, GC).norm(dim=2)
        orders.append(torch.argsort(bnd[None, :] * xn, dim=1, descending=True))
    if static_order is not None:
        orders.append(static_order[None, :].expand(tb, -1))
    for oi, order in enumerate(orders):
        Cs = torch.gather(C, 1, order[:, :, None].expand(-1, -1, C.shape[2]))
        S = torch.cumsum(Cs, dim=1)
        del Cs
        E = r[:, None, :] - S
        err = torch.cat([r.norm(dim=1, keepdim=True), E.norm(dim=2)], dim=1) / ynorm[:, None]
        del E
        out.append(err)
        if oi == 0 and need_S:
            S_or = S
        else:
            del S
    e_on = out[1] if bnd is not None else None
    e_st = out[-1] if static_order is not None else None
    return out[0], e_on, e_st, S_or


def static_orders(Rg: torch.Tensor, Xcal: torch.Tensor, tb: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Static (token-independent) block orders from CALIBRATION activations.

    Column blocks by mean ||R_g x_g||^2, row blocks by mean ||(R x)_rows||^2 (descending)."""
    G, n_out, _ = Rg.shape
    ec = torch.zeros(G, dtype=torch.float64)
    er = torch.zeros(n_out // GC, dtype=torch.float64)
    T = Xcal.shape[0]
    for s0 in range(0, T, tb):
        X = Xcal[s0:s0 + tb]
        n = X.shape[0]
        C = torch.bmm(Rg, X.view(n, G, GC).permute(1, 2, 0))          # [G, n_out, n]
        ec += (C.double() ** 2).sum(dim=(1, 2))
        rr = C.sum(0).T                                                 # [n, n_out] = R x
        er += (rr.double().view(n, -1, GC) ** 2).sum(dim=(0, 2))
        del C
    return torch.argsort(ec, descending=True), torch.argsort(er, descending=True)


def row_static_err(r: torch.Tensor, ynorm: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    tb, n_out = r.shape
    e2 = (r.view(tb, n_out // GC, GC) ** 2).sum(2)[:, order]
    rem = torch.clamp(e2.sum(1, keepdim=True) - torch.cumsum(e2, dim=1), min=0)
    return torch.cat([e2.sum(1, keepdim=True), rem], dim=1).sqrt() / ynorm[:, None]


def k90(err: np.ndarray, eps: float, q: float = 0.9) -> int | None:
    """Smallest fixed k (same for all tokens) with the q-quantile of err[:, k] <= eps."""
    qs = np.quantile(err, q, axis=0)
    ok = np.flatnonzero(qs <= eps)
    return int(ok[0]) if ok.size else None


def row_analysis(r: torch.Tensor, ynorm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tb, n_out = r.shape
    Gr = n_out // GC
    e2 = (r.view(tb, Gr, GC) ** 2).sum(2)                  # [tb, Gr]
    s, order = torch.sort(e2, dim=1, descending=True)
    rem = e2.sum(1, keepdim=True) - torch.cumsum(s, dim=1)
    rem = torch.clamp(rem, min=0)
    err = torch.cat([e2.sum(1, keepdim=True), rem], dim=1).sqrt() / ynorm[:, None]
    return err, order


def k_needed(err: torch.Tensor, eps: float) -> torch.Tensor:
    """Smallest k with err[:, k] <= eps (G if never; the last column is ~0 so always reachable)."""
    ok = err <= eps
    ok[:, -1] = True
    return torch.argmax(ok.to(torch.int8), dim=1)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


# ------------------------------------------------------------------------------------------
# Per layer
# ------------------------------------------------------------------------------------------
def run_layer(il: int, cal_dir: Path, ho_dir: Path, ranks: list[int], comp_ranks: list[int],
              diag_ranks: list[int], tb: int = 48, max_heldout: int = 1024,
              fit_cache: Path | None = None, fit_only: bool = False) -> dict[str, Any]:
    t_layer = time.time()
    res: dict[str, Any] = {"layer": il, "started_utc": _now()}
    names = dc.ffn_names(il)
    Wn = {m: dc.load_matrix(MAP31, names[m]) for m in ("gate", "up", "down")}
    info = {m: dc.tensor_info(MAP31, names[m]) for m in ("gate", "up", "down")}
    res["tensors"] = {m: {"name": names[m], "type": info[m].type, "shape_ne": list(info[m].shape),
                          "n_bytes": info[m].n_bytes, "offset": info[m].offset} for m in names}
    W = {m: torch.from_numpy(Wn[m]) for m in Wn}
    cal = load_acts(cal_dir, il)
    if fit_only:
        Xc = {"gate": torch.from_numpy(cal["ffn_norm"]), "down": torch.from_numpy(cal["ffn_geglu"])}
        Xc["up"] = Xc["gate"]
        fits = {}
        for kind in ("whiten", "diag"):
            for m in ("gate", "up", "down"):
                tt = time.time()
                rmax = max(ranks) if kind == "whiten" else max(diag_ranks)
                cp = fit_cache / f"L{il}.{kind}.{m}.r{rmax}.T{Xc[m].shape[0]}.pt"
                if cp.is_file():
                    continue
                Uk, sig = fit_U(W[m], Xc[m], kind, rmax)
                torch.save({"U": Uk, "sig": sig, "seconds": time.time() - tt}, cp)
                log(f"L{il}: fit-only {kind}/{m} {time.time() - tt:.1f}s -> {cp}")
        return {"layer": il, "fit_only": True}
    ho = load_acts(ho_dir, il)
    Xc = {"gate": torch.from_numpy(cal["ffn_norm"]), "down": torch.from_numpy(cal["ffn_geglu"])}
    Xc["up"] = Xc["gate"]
    T_all = ho["ffn_norm"].shape[0]
    gen_idx = np.flatnonzero(ho["phase"] == 1)
    pre_idx = np.flatnonzero(ho["phase"] != 1)
    if max_heldout and T_all > max_heldout:
        n_pre = max(0, max_heldout - gen_idx.size)
        step = max(1, int(math.ceil(pre_idx.size / max(n_pre, 1))))
        keep = np.sort(np.concatenate([gen_idx[:max_heldout], pre_idx[::step][:n_pre]]))
    else:
        keep = np.arange(T_all)
    for k in ("ffn_norm", "ffn_geglu", "ffn_out"):
        ho[k] = ho[k][keep]
    ho["phase"] = ho["phase"][keep]
    Xh = torch.from_numpy(ho["ffn_norm"])
    Hh = torch.from_numpy(ho["ffn_geglu"])
    Oh = torch.from_numpy(ho["ffn_out"])
    Th = Xh.shape[0]
    res["tokens"] = {"calibration": int(Xc["gate"].shape[0]), "heldout": int(Th), "heldout_captured": int(T_all),
                     "heldout_subsample": "all generated tokens + every k-th prefill token (deterministic), capped at --max-heldout",
                     "heldout_gen": int((ho["phase"] == 1).sum()), "heldout_prefill": int((ho["phase"] == 0).sum()),
                     "dropped_nonfinite": {"cal": cal["n_dropped_nonfinite"], "heldout": ho["n_dropped_nonfinite"]}}
    log(f"L{il}: cal {res['tokens']['calibration']} tok, held-out {Th} tok")

    # ---- runtime floor: llama.cpp output vs the float32 reference (held-out) ----
    with torch.no_grad():
        g_ref = Xh @ W["gate"].T
        u_ref = Xh @ W["up"].T
        h_ref = gelu(g_ref) * u_ref
        y_ref = h_ref @ W["down"].T
        yn = y_ref.norm(dim=1)
        res["runtime_floor"] = {
            "what": "llama.cpp a7b94df (Docker, CPU repack kernels, Q8 activations) vs float32 math on the dequantised Q4_0 weights",
            "geglu_captured_vs_f32_from_captured_ffn_norm": stats(((h_ref - Hh).norm(dim=1) / h_ref.norm(dim=1)).numpy()),
            "ffn_out_captured_vs_f32_down_of_captured_geglu": stats(((Hh @ W["down"].T - Oh).norm(dim=1) / Oh.norm(dim=1)).numpy()),
            "ffn_out_captured_vs_f32_composed_from_captured_ffn_norm": stats(((y_ref - Oh).norm(dim=1) / yn).numpy()),
        }
        del g_ref, u_ref
    log(f"L{il}: runtime floor (composed) median {res['runtime_floor']['ffn_out_captured_vs_f32_composed_from_captured_ffn_norm']['median']:.4f}")

    q4b = {m: info[m].n_bytes for m in W}
    ffn_q4 = sum(q4b.values())
    res["bytes"] = {"q4_0_per_matrix": q4b, "q4_0_ffn": ffn_q4}

    # ---- uniform baselines (isolated + composed) ----
    t0 = time.time()
    imat = {"gate": (Xc["gate"] ** 2).mean(0).numpy(), "down": (Xc["down"] ** 2).mean(0).numpy()}
    imat["up"] = imat["gate"]
    unif: dict[str, Any] = {}
    Wq_all: dict[str, dict[str, torch.Tensor]] = {}
    for qn in ("q3_k_imatrix", "q3_absmax", "q2_absmax"):
        Wq_all[qn] = {}
        unif[qn] = {"bytes_per_weight": {"q3_k_imatrix": B_Q3_K, "q3_absmax": B_Q3, "q2_absmax": B_Q2}[qn], "matrices": {}}
        for m in ("gate", "up", "down"):
            if qn == "q3_k_imatrix":
                Wq, ver = q3k_matrix(Wn[m], imat[m])
                unif[qn]["matrices"].setdefault(m, {})["format_check"] = ver
            else:
                Wq, _ = dc.group_quant(Wn[m], 3 if qn == "q3_absmax" else 2, 32)
            Wq_all[qn][m] = torch.from_numpy(Wq)
        with torch.no_grad():
            gq = Xh @ Wq_all[qn]["gate"].T
            uq = Xh @ Wq_all[qn]["up"].T
            iso = {
                "gate": ((gq - Xh @ W["gate"].T).norm(dim=1) / (Xh @ W["gate"].T).norm(dim=1)).numpy(),
                "up": ((uq - Xh @ W["up"].T).norm(dim=1) / (Xh @ W["up"].T).norm(dim=1)).numpy(),
                "down": ((Hh @ Wq_all[qn]["down"].T - Hh @ W["down"].T).norm(dim=1) / (Hh @ W["down"].T).norm(dim=1)).numpy(),
            }
            yq = (gelu(gq) * uq) @ Wq_all[qn]["down"].T
            ffn_err = ((yq - y_ref).norm(dim=1) / yn).numpy()
        for m in iso:
            unif[qn]["matrices"].setdefault(m, {})["rel_err"] = stats(iso[m])
        unif[qn]["ffn_rel_err"] = stats(ffn_err)
        unif[qn]["ffn_bytes"] = int(sum(round(W[m].numel() * unif[qn]["bytes_per_weight"]) for m in W))
        log(f"L{il}: {qn}: ffn err mean {unif[qn]['ffn_rel_err']['mean']:.4f} "
            f"(gate {unif[qn]['matrices']['gate']['rel_err']['mean']:.4f}, up {unif[qn]['matrices']['up']['rel_err']['mean']:.4f}, "
            f"down {unif[qn]['matrices']['down']['rel_err']['mean']:.4f})")
    res["uniform"] = unif
    res["uniform_seconds"] = time.time() - t0
    del Wq_all

    # ---- fits ----
    fits: dict[str, Any] = {}
    U: dict[tuple[str, str], torch.Tensor] = {}
    t0 = time.time()
    for kind in ("whiten", "diag"):
        for m in ("gate", "up", "down"):
            tt = time.time()
            rmax = max(ranks) if kind == "whiten" else max(diag_ranks)
            cp = fit_cache / f"L{il}.{kind}.{m}.r{rmax}.T{Xc[m].shape[0]}.pt" if fit_cache else None
            if cp is not None and cp.is_file():
                ck = torch.load(cp, weights_only=False)  # own cache file (U tensor + numpy sigma)
                Uk, sig, fsec = ck["U"], ck["sig"], ck["seconds"]
            else:
                Uk, sig = fit_U(W[m], Xc[m], kind, rmax)
                fsec = time.time() - tt
                if cp is not None:
                    torch.save({"U": Uk, "sig": sig, "seconds": fsec}, cp)
            U[(kind, m)] = Uk
            # calibration energy captured: fraction of ||W X||^2 (whiten: incl. ridge) explained
            e = sig.astype(np.float64) ** 2
            fits.setdefault(kind, {})[m] = {"seconds": fsec,
                                            "energy_frac_at_rank": {str(r): float(e[:r].sum() / e.sum()) for r in ranks}}
            log(f"L{il}: fit {kind}/{m} {time.time() - tt:.1f}s energy@{ranks} = "
                + ", ".join(f"{fits[kind][m]['energy_frac_at_rank'][str(r)]:.3f}" for r in ranks))
    res["fits"] = fits
    res["fit_seconds"] = time.time() - t0
    if fit_only:
        return res

    # ---- isolated per-matrix evaluation (held-out); composed FFN for the whiten ranks in comp_ranks ----
    iso_res: dict[str, Any] = {}
    comp: dict[str, Any] = {}
    gen_mask = ho["phase"] == 1
    Xcal_sub = {m: Xc[m][::4].contiguous() for m in ("gate", "up", "down")}
    res["static_selection"] = {"what": "token-independent block order by mean calibration energy (col: ||R_g x_g||^2, row: ||(R x)_rows||^2)",
                               "calibration_tokens_used": int(Xcal_sub["gate"].shape[0])}
    t0 = time.time()
    t_comp = 0.0
    for kind in ("whiten", "diag"):
        for r in (ranks if kind == "whiten" else diag_ranks):
            store: dict[str, dict[str, Any]] = {}
            for m in ("gate", "up", "down"):
                Wm = W[m]
                n_out, n_in = Wm.shape
                A_f = U[(kind, m)][:, :r].contiguous()
                B_f = A_f.T @ Wm
                A = q8(A_f)
                B = q8(B_f)
                base_bytes = int(round((A.numel() + B.numel()) * B_Q8))
                R = Wm - A @ B
                Gc = n_in // GC
                Rg = R.view(n_out, Gc, GC).permute(1, 0, 2).contiguous()
                del R
                bnd = torch.linalg.matrix_norm(Rg, ord=2)                      # [Gc] spectral norms
                st_col, st_row = static_orders(Rg, Xcal_sub[m])
                Xin = Xh if m in ("gate", "up") else Hh
                colb = n_out * GC * B_Q4_0
                rowb = GC * n_in * B_Q4_0
                acc_or, acc_on, acc_row, acc_cst, acc_rst = [], [], [], [], []
                base_err, fp_base_err, recon, q8_first = [], [], [], []
                for s0 in range(0, Th, tb):
                    X = Xin[s0:s0 + tb]
                    y = X @ Wm.T
                    ynorm = y.norm(dim=1)
                    yb = (X @ B.T) @ A.T
                    rr = y - yb
                    e_or, e_on, e_cst, _ = col_analysis(Rg, X, rr, ynorm, bnd, static_order=st_col)
                    e_row, _ = row_analysis(rr, ynorm)
                    acc_cst.append(e_cst)
                    acc_rst.append(row_static_err(rr, ynorm, st_row))
                    acc_or.append(e_or)
                    acc_on.append(e_on)
                    acc_row.append(e_row)
                    base_err.append(rr.norm(dim=1) / ynorm)
                    if s0 == 0:
                        ybf = (X @ B_f.T) @ A_f.T
                        fp_base_err.append(((y - ybf).norm(dim=1) / ynorm))
                        q8_first.append(rr.norm(dim=1) / ynorm)
                        # exactness of base + all column blocks (runtime form A(B x) + sum_g R_g x_g)
                        Xg = X.view(X.shape[0], Gc, GC).permute(1, 2, 0)
                        full = yb + torch.bmm(Rg, Xg).sum(0).T
                        recon.append(((full - y).norm(dim=1) / ynorm))
                e_or_raw = torch.cat(acc_or).numpy()
                e_or = np.minimum.accumulate(e_or_raw, axis=1)   # oracle may stop early: best prefix
                e_on = torch.cat(acc_on).numpy()
                e_row = torch.cat(acc_row).numpy()
                e_cst = torch.cat(acc_cst).numpy()
                e_rst = torch.cat(acc_rst).numpy()
                be = torch.cat(base_err).numpy()
                Gr = n_out // GC
                mono = float((np.diff(e_or_raw, axis=1) <= 1e-7).all(axis=1).mean())
                hump = float((e_or_raw.max(axis=1) > e_or_raw[:, 0] * 1.0001).mean())
                ent: dict[str, Any] = {
                    "col_oracle_monotone_token_frac": mono,
                    "col_oracle_tokens_where_partial_reading_exceeds_base_err_frac": hump,
                    "col_greedy_raw_curve_max_over_base_err_mean": float((e_or_raw.max(axis=1) / e_or_raw[:, 0]).mean()),
                    "kind": kind, "rank": r, "matrix": m, "base_bytes": base_bytes,
                    "base_frac_of_q4_0": base_bytes / q4b[m], "base_rel_err": stats(be),
                    "base_rel_err_gen_tokens": stats(be[gen_mask]),
                    "base_rel_err_fp32_factors_first_batch": stats(torch.cat(fp_base_err).numpy()),
                    "base_rel_err_q8_factors_first_batch": stats(torch.cat(q8_first).numpy()),
                    "reconstruction_rel_err_all_blocks_first_batch": stats(torch.cat(recon).numpy()),
                    "n_col_blocks": Gc, "n_row_blocks": Gr, "col_block_bytes": colb, "row_block_bytes": rowb,
                    "meta_bytes_resident": Gc * 12, "thresholds": {},
                }
                for eps in EPS_MAT:
                    kc = (e_or <= eps); kc[:, -1] = True; kc = kc.argmax(1)
                    ko = (e_on <= eps); ko[:, -1] = True; ko = ko.argmax(1)
                    kr = (e_row <= eps); kr[:, -1] = True; kr = kr.argmax(1)
                    ent["thresholds"][f"{eps:g}"] = {
                        "col_oracle_frac": stats(kc / Gc), "col_online_frac": stats(ko / Gc),
                        "row_oracle_frac": stats(kr / Gr),
                        "col_oracle_frac_gen_tokens_mean": float((kc[gen_mask] / Gc).mean()) if gen_mask.any() else None,
                        "col_static_fixed_frac_for_90pct_tokens": (None if k90(e_cst, eps) is None else k90(e_cst, eps) / Gc),
                        "row_static_fixed_frac_for_90pct_tokens": (None if k90(e_rst, eps) is None else k90(e_rst, eps) / Gr),
                        "col_oracle_total_bytes_mean": float(base_bytes + Gc * 12 + (kc * (colb + 2)).mean()),
                        "row_oracle_total_bytes_mean": float(base_bytes + (kr * (rowb + 2)).mean()),
                    }
                eq = {}
                for qn, bpw in (("q3_k_imatrix", B_Q3_K), ("q3_absmax", B_Q3), ("q2_absmax", B_Q2)):
                    budget = Wm.numel() * bpw
                    kcb = int(max(0, min(Gc, math.floor((budget - base_bytes - Gc * 12) / (colb + 2)))))
                    krb = int(max(0, min(Gr, math.floor((budget - base_bytes) / (rowb + 2)))))
                    feasible = budget >= base_bytes
                    eq[qn] = {"budget_bytes": budget, "base_fits": bool(feasible), "k_col": kcb, "k_row": krb,
                              "col_oracle_err": stats(e_or[:, kcb]) if feasible else None,
                              "col_oracle_err_exactly_k_blocks": stats(e_or_raw[:, kcb]) if feasible else None,
                              "col_online_err": stats(e_on[:, kcb]) if feasible else None,
                              "row_oracle_err": stats(e_row[:, krb]) if feasible else None,
                              "col_static_err": stats(e_cst[:, kcb]) if feasible else None,
                              "row_static_err": stats(e_rst[:, krb]) if feasible else None,
                              "uniform_err": unif[qn]["matrices"][m]["rel_err"]}
                ent["equal_bytes"] = eq
                fr = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                ent["curve_col_oracle_mean_err"] = {f"{f:g}": float(e_or[:, int(round(f * Gc))].mean()) for f in fr}
                ent["curve_col_online_mean_err"] = {f"{f:g}": float(e_on[:, int(round(f * Gc))].mean()) for f in fr}
                ent["curve_row_oracle_mean_err"] = {f"{f:g}": float(e_row[:, int(round(f * Gr))].mean()) for f in fr}
                ent["curve_col_static_mean_err"] = {f"{f:g}": float(e_cst[:, int(round(f * Gc))].mean()) for f in fr}
                ent["curve_row_static_mean_err"] = {f"{f:g}": float(e_rst[:, int(round(f * Gr))].mean()) for f in fr}
                iso_res[f"{kind}/r{r}/{m}"] = ent
                if kind == "whiten" and r in comp_ranks:
                    store[m] = {"A": A, "B": B, "Rg": Rg, "bnd": bnd, "base_bytes": base_bytes,
                                "st_col": st_col, "st_row": st_row}
                else:
                    del Rg
                e01 = ent["thresholds"]["0.01"]
                q3e = eq["q3_k_imatrix"]["col_oracle_err"]
                log(f"L{il}: {kind} r{r} {m}: base {base_bytes / q4b[m]:.2f}xQ4 err {ent['base_rel_err']['mean']:.3f} | "
                    f"@1e-2 col-oracle {e01['col_oracle_frac']['mean']:.3f} online {e01['col_online_frac']['mean']:.3f} "
                    f"row {e01['row_oracle_frac']['mean']:.3f} | @Q3K bytes col {q3e['mean'] if q3e else float('nan'):.4f} "
                    f"row {eq['q3_k_imatrix']['row_oracle_err']['mean'] if eq['q3_k_imatrix']['row_oracle_err'] else float('nan'):.4f} "
                    f"col-static {eq['q3_k_imatrix']['col_static_err']['mean'] if eq['q3_k_imatrix']['col_static_err'] else float('nan'):.4f} "
                    f"vs Q3K {unif['q3_k_imatrix']['matrices'][m]['rel_err']['mean']:.4f}")
            if store:
                tc = time.time()
                comp[f"r{r}"] = composed_rank(il, r, store, W, Xh, y_ref, unif, ffn_q4, gen_mask, tb)
                t_comp += time.time() - tc
                del store
    res["isolated"] = iso_res
    res["composed"] = comp
    res["isolated_seconds"] = time.time() - t0 - t_comp
    res["composed_seconds"] = t_comp
    res["seconds"] = time.time() - t_layer
    res["finished_utc"] = _now()
    return res


def _row_mask(order: torch.Tensor, k: int, n_out: int) -> torch.Tensor:
    m_ = torch.zeros(n_out // GC, dtype=torch.bool)
    if k > 0:
        m_[order[:k]] = True
    return m_.repeat_interleave(GC)[None, :]


def _masked_col_apply(Rg: torch.Tensor, X: torch.Tensor, order: torch.Tensor, k: int) -> torch.Tensor:
    """sum_{g in order[:, :k]} R_g x_g for every token (the online / fixed-k path)."""
    tb = X.shape[0]
    G = Rg.shape[0]
    if k <= 0:
        return torch.zeros(tb, Rg.shape[1])
    m_ = torch.zeros(tb, G, dtype=torch.bool)
    m_.scatter_(1, order[:, :k], True)
    Xm = X.view(tb, G, GC) * m_[:, :, None]
    return torch.bmm(Rg, Xm.permute(1, 2, 0)).sum(0).T


def composed_rank(il: int, r: int, st: dict[str, dict[str, Any]], W: dict[str, torch.Tensor], Xh: torch.Tensor,
                  y_ref: torch.Tensor, unif: dict[str, Any], ffn_q4: int, gen_mask: np.ndarray, tb: int) -> dict[str, Any]:
    """FFN output error vs bytes with every matrix approximated (captured layer input x, held-out)."""
    cg, cu, cd = st["gate"], st["up"], st["down"]
    Th = Xh.shape[0]
    base_ffn = cg["base_bytes"] + cu["base_bytes"] + cd["base_bytes"]
    Gcg = W["gate"].shape[1] // GC
    Gcd = W["down"].shape[1] // GC
    Grg = W["gate"].shape[0] // GC
    Grd = W["down"].shape[0] // GC
    colb_g = W["gate"].shape[0] * GC * B_Q4_0
    colb_d = W["down"].shape[0] * GC * B_Q4_0
    rowb_g = GC * W["gate"].shape[1] * B_Q4_0
    rowb_d = GC * W["down"].shape[1] * B_Q4_0
    meta = (2 * Gcg + Gcd) * 12
    per = {f"{e:g}": {"col": {"err": [], "bytes": []}, "row": {"err": [], "bytes": []}} for e in EPS_MAT}
    eqk: dict[str, Any] = {}
    eq_points: dict[str, Any] = {}
    for qn in ("q3_k_imatrix", "q3_absmax", "q2_absmax"):
        bud = unif[qn]["ffn_bytes"]
        avail = bud - base_ffn - meta
        f = max(0.0, min(1.0, avail / ffn_q4)) if avail > 0 else 0.0
        eqk[qn] = {"budget": bud, "residual_frac": f, "base_fits": avail >= 0,
                   "kg": int(math.floor(f * Gcg)), "kd": int(math.floor(f * Gcd)),
                   "rg": int(math.floor(f * Grg)), "rd": int(math.floor(f * Grd))}
        eq_points[qn] = {"col_err": [], "col_online_err": [], "row_err": [], "col_static_err": [], "row_static_err": []}
    fixed = {f"{f:g}": {"col_static": [], "row_static": []} for f in FIX_F}
    ar = torch.arange
    for s0 in range(0, Th, tb):
        X = Xh[s0:s0 + tb]
        n = X.shape[0]
        yt = y_ref[s0:s0 + tb]
        ytn = yt.norm(dim=1)
        outs = {}
        for nm, cc in (("gate", cg), ("up", cu)):
            y = X @ W[nm].T
            yb = (X @ cc["B"].T) @ cc["A"].T
            rr = y - yb
            ynorm = y.norm(dim=1)
            e_or, _, _, S = col_analysis(cc["Rg"], X, rr, ynorm, None, need_S=True)
            e_row, rorder = row_analysis(rr, ynorm)
            outs[nm] = (yb, rr, e_or, S, e_row, torch.argsort(rorder, dim=1))
        for eps in EPS_MAT:
            key = f"{eps:g}"
            # column organisation, oracle per matrix
            hs, byt = [], torch.zeros(n, dtype=torch.float64)
            for nm in ("gate", "up"):
                yb, rr, e_or, S, e_row, rpos = outs[nm]
                k = k_needed(e_or, eps)
                add = S[ar(n), torch.clamp(k - 1, min=0)] * (k > 0)[:, None]
                hs.append(yb + add)
                byt += k.double() * (colb_g + 2)
            h = gelu(hs[0]) * hs[1]
            yd = h @ W["down"].T
            ybd = (h @ cd["B"].T) @ cd["A"].T
            rd = yd - ybd
            e_d, _, _, Sd = col_analysis(cd["Rg"], h, rd, yd.norm(dim=1), None, need_S=True)
            kd = k_needed(e_d, eps)
            yfin = ybd + Sd[ar(n), torch.clamp(kd - 1, min=0)] * (kd > 0)[:, None]
            byt += kd.double() * (colb_d + 2)
            per[key]["col"]["err"].append(((yfin - yt).norm(dim=1) / ytn).numpy())
            per[key]["col"]["bytes"].append(byt.numpy() + base_ffn + meta)
            del Sd
            # row organisation, exact optimal per matrix
            hs, byt = [], torch.zeros(n, dtype=torch.float64)
            for nm in ("gate", "up"):
                yb, rr, e_or, S, e_row, rpos = outs[nm]
                k = k_needed(e_row, eps)
                sel = (rpos < k[:, None]).repeat_interleave(GC, dim=1)
                hs.append(yb + rr * sel)
                byt += k.double() * (rowb_g + 2)
            h = gelu(hs[0]) * hs[1]
            yd = h @ W["down"].T
            ybd = (h @ cd["B"].T) @ cd["A"].T
            rd = yd - ybd
            e_rd, rord = row_analysis(rd, yd.norm(dim=1))
            kd = k_needed(e_rd, eps)
            sel = (torch.argsort(rord, dim=1) < kd[:, None]).repeat_interleave(GC, dim=1)
            yfin = ybd + rd * sel
            byt += kd.double() * (rowb_d + 2)
            per[key]["row"]["err"].append(((yfin - yt).norm(dim=1) / ytn).numpy())
            per[key]["row"]["bytes"].append(byt.numpy() + base_ffn)
        # equal-byte points: the same residual fraction on all three matrices
        for qn, q in eqk.items():
            if not q["base_fits"]:
                continue
            for mode in ("col", "col_online", "row", "col_static", "row_static"):
                hs = []
                for nm, cc in (("gate", cg), ("up", cu)):
                    yb, rr, e_or, S, e_row, rpos = outs[nm]
                    if mode == "row":
                        hs.append(yb + rr * (rpos < q["rg"]).repeat_interleave(GC, dim=1))
                    elif mode == "col_static":
                        hs.append(yb + _masked_col_apply(cc["Rg"], X, cc["st_col"][None, :].expand(n, -1), q["kg"]))
                    elif mode == "row_static":
                        hs.append(yb + rr * _row_mask(cc["st_row"], q["rg"], rr.shape[1]))
                    elif mode == "col":
                        kb = torch.argmin(e_or[:, :q["kg"] + 1], dim=1)
                        hs.append(yb + S[torch.arange(n), torch.clamp(kb - 1, min=0)] * (kb > 0)[:, None])
                    else:
                        xn = X.view(n, Gcg, GC).norm(dim=2)
                        order = torch.argsort(cc["bnd"][None, :] * xn, dim=1, descending=True)
                        hs.append(yb + _masked_col_apply(cc["Rg"], X, order, q["kg"]))
                h = gelu(hs[0]) * hs[1]
                yd = h @ W["down"].T
                ybd = (h @ cd["B"].T) @ cd["A"].T
                rd = yd - ybd
                if mode == "row":
                    _, rord = row_analysis(rd, yd.norm(dim=1))
                    yfin = ybd + rd * (torch.argsort(rord, dim=1) < q["rd"]).repeat_interleave(GC, dim=1)
                elif mode == "col_static":
                    yfin = ybd + _masked_col_apply(cd["Rg"], h, cd["st_col"][None, :].expand(n, -1), q["kd"])
                elif mode == "row_static":
                    yfin = ybd + rd * _row_mask(cd["st_row"], q["rd"], rd.shape[1])
                elif mode == "col":
                    e_dd, _, _, Sd = col_analysis(cd["Rg"], h, rd, yd.norm(dim=1), None, need_S=True)
                    kb = torch.argmin(e_dd[:, :q["kd"] + 1], dim=1)
                    yfin = ybd + Sd[torch.arange(n), torch.clamp(kb - 1, min=0)] * (kb > 0)[:, None]
                    del Sd
                else:
                    xn = h.view(n, Gcd, GC).norm(dim=2)
                    order = torch.argsort(cd["bnd"][None, :] * xn, dim=1, descending=True)
                    yfin = ybd + _masked_col_apply(cd["Rg"], h, order, q["kd"])
                key2 = {"col": "col_err", "col_online": "col_online_err", "row": "row_err",
                        "col_static": "col_static_err", "row_static": "row_static_err"}[mode]
                eq_points[qn][key2].append(((yfin - yt).norm(dim=1) / ytn).numpy())
        # static selection at fixed fractions (f = 0 is the base alone)
        for fkey in fixed:
            f = float(fkey)
            for mode in ("col_static", "row_static"):
                hs = []
                for nm, cc in (("gate", cg), ("up", cu)):
                    yb, rr = outs[nm][0], outs[nm][1]
                    if mode == "col_static":
                        hs.append(yb + _masked_col_apply(cc["Rg"], X, cc["st_col"][None, :].expand(n, -1), int(math.floor(f * Gcg))))
                    else:
                        hs.append(yb + rr * _row_mask(cc["st_row"], int(math.floor(f * Grg)), rr.shape[1]))
                h = gelu(hs[0]) * hs[1]
                yd = h @ W["down"].T
                ybd = (h @ cd["B"].T) @ cd["A"].T
                rd = yd - ybd
                if mode == "col_static":
                    yfin = ybd + _masked_col_apply(cd["Rg"], h, cd["st_col"][None, :].expand(n, -1), int(math.floor(f * Gcd)))
                else:
                    yfin = ybd + rd * _row_mask(cd["st_row"], int(math.floor(f * Grd)), rd.shape[1])
                fixed[fkey][mode].append(((yfin - yt).norm(dim=1) / ytn).numpy())
        del outs
    cres: dict[str, Any] = {"rank": r, "base_bytes_ffn": base_ffn, "base_frac_of_q4_0_ffn": base_ffn / ffn_q4,
                            "meta_bytes_resident": meta, "per_matrix_eps": {}}
    for key, v in per.items():
        ent = {}
        for org in ("col", "row"):
            err = np.concatenate(v[org]["err"])
            byt = np.concatenate(v[org]["bytes"])
            ent[org] = {"ffn_rel_err": stats(err), "ffn_rel_err_gen_tokens": stats(err[gen_mask]),
                        "total_bytes": stats(byt), "total_frac_of_q4_0": float(byt.mean() / ffn_q4),
                        "residual_frac_of_q4_0": float((byt.mean() - base_ffn - (meta if org == "col" else 0)) / ffn_q4),
                        "frac_tokens_ffn_err_le": {f"{t:g}": float((err <= t).mean()) for t in FFN_TARGETS}}
        cres["per_matrix_eps"][key] = ent
    cres["equal_bytes"] = {}
    for qn, q in eqk.items():
        e = {"budget_bytes": q["budget"], "base_fits": q["base_fits"], "residual_frac": q["residual_frac"],
             "uniform_ffn_err": unif[qn]["ffn_rel_err"]}
        if q["base_fits"]:
            for k2 in ("col_err", "col_online_err", "row_err", "col_static_err", "row_static_err"):
                arr = np.concatenate(eq_points[qn][k2])
                e[k2] = stats(arr)
                e[k2 + "_gen_tokens"] = stats(arr[gen_mask])
        cres["equal_bytes"][qn] = e
    cres["static_fixed_fraction"] = {}
    for fkey, v in fixed.items():
        ent = {"residual_frac": float(fkey), "total_frac_of_q4_0": (base_ffn + float(fkey) * ffn_q4) / ffn_q4}
        for mode in ("col_static", "row_static"):
            err = np.concatenate(v[mode])
            ent[mode] = {"ffn_rel_err": stats(err), "ffn_rel_err_gen_tokens": stats(err[gen_mask]),
                         "frac_tokens_ffn_err_le": {f"{t:g}": float((err <= t).mean()) for t in FFN_TARGETS}}
        cres["static_fixed_fraction"][fkey] = ent
    q3 = cres["equal_bytes"]["q3_k_imatrix"]
    log(f"L{il}: composed r{r}: base {cres['base_frac_of_q4_0_ffn']:.2f}xQ4 | @Q3K bytes col "
        f"{q3.get('col_err', {}).get('mean', float('nan')):.4f} row {q3.get('row_err', {}).get('mean', float('nan')):.4f} "
        f"online {q3.get('col_online_err', {}).get('mean', float('nan')):.4f} col-static {q3.get('col_static_err', {}).get('mean', float('nan')):.4f} "
        f"row-static {q3.get('row_static_err', {}).get('mean', float('nan')):.4f} vs Q3K {q3['uniform_ffn_err']['mean']:.4f} | "
        f"base alone {cres['static_fixed_fraction']['0']['col_static']['ffn_rel_err']['mean']:.4f}")
    return cres


# ------------------------------------------------------------------------------------------
# Runtime (prototype, torch fp32, batch 1)
# ------------------------------------------------------------------------------------------
def time_it(fn, reps: int = 30) -> dict[str, float]:
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return {"median_ms": 1e3 * float(np.median(ts)), "p90_ms": 1e3 * float(np.quantile(ts, 0.9))}


def runtime_bench(il: int, ho_dir: Path, ranks: list[int]) -> dict[str, Any]:
    names = dc.ffn_names(il)
    out: dict[str, Any] = {"layer": il, "threads": torch.get_num_threads(),
                           "what": "torch fp32 prototype on the desktop CPU, batch 1 (one decode token); NOT ggml Q4_0 kernels"}
    ho = load_acts(ho_dir, il)
    for m in ("gate", "down"):
        W = torch.from_numpy(dc.load_matrix(MAP31, names[m]))
        n_out, n_in = W.shape
        x = torch.from_numpy(ho["ffn_norm" if m == "gate" else "ffn_geglu"][5]).clone()
        Gc = n_in // GC
        ent: dict[str, Any] = {"shape": [n_out, n_in]}
        ent["dense_fp32_matvec"] = time_it(lambda: W @ x)
        Wcb = W.view(n_out, Gc, GC).permute(1, 0, 2).contiguous()          # column-block-major
        nrm = torch.rand(Gc)
        def sel(k):
            xn = x.view(Gc, GC).norm(dim=1)
            return torch.topk(nrm * xn, k).indices
        ent["selector_norm_bound_topk"] = {f"{f:g}": time_it(lambda f=f: sel(max(1, int(f * Gc)))) for f in (0.1, 0.5)}
        xg = x.view(Gc, GC)
        perm = torch.randperm(Gc)

        def resid(idx):
            y = torch.zeros(n_out)
            for g in idx.tolist():
                y.addmv_(Wcb[g], xg[g])          # Wcb[g] is a contiguous view: no copy
            return y

        for f in (0.1, 0.25, 0.5, 1.0):
            k = max(1, int(round(f * Gc)))
            idx = perm[:k]
            ent.setdefault("residual_blockloop_matvec", {})[f"{f:g}"] = time_it(lambda idx=idx: resid(idx))
        ent["note"] = ("fp32 data: the dense matvec reads 4 B/weight, the base reads fp32 factors, the residual loop "
                       "reads fp32 column blocks; a Q4_0/Q8 implementation would move 0.5625 / 1.0625 B/weight")
        for r in ranks:
            A = torch.randn(n_out, r)
            B = torch.randn(r, n_in)
            ent.setdefault("base_lowrank_matvec", {})[str(r)] = time_it(lambda A=A, B=B: A @ (B @ x))
        out[m] = ent
        del W, Wcb
    return out


def cpu_load_sample(seconds: float = 3.0) -> float | None:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=seconds))
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cal", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--layers", default="2,30,57")
    ap.add_argument("--ranks", default="128,256,512,1024,2048")
    ap.add_argument("--composed-ranks", default="512,1024")
    ap.add_argument("--diag-ranks", default="512")
    ap.add_argument("--max-heldout", type=int, default=1024)
    ap.add_argument("--out", default="results/dense")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tb", type=int, default=48)
    ap.add_argument("--runtime-only", action="store_true")
    ap.add_argument("--fit-cache", default=None, help="directory for cached base fits (U matrices)")
    ap.add_argument("--fit-only", action="store_true", help="only compute + cache the calibration fits")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fit_cache:
        Path(a.fit_cache).mkdir(parents=True, exist_ok=True)
    ranks = [int(x) for x in a.ranks.split(",")]
    comp_ranks = [int(x) for x in a.composed_ranks.split(",")]
    host = {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "threads": a.threads}
    if a.runtime_only:
        load = cpu_load_sample()
        rb = runtime_bench(int(a.layers.split(",")[0]), Path(a.heldout), ranks)
        rb["bg_cpu_percent_before"] = load
        rb["bg_cpu_percent_after"] = cpu_load_sample()
        rb["host"] = host
        rb["created_utc"] = _now()
        (out / "runtime.json").write_text(json.dumps(rb, indent=1), encoding="utf-8")
        print(json.dumps(rb, indent=1))
        return 0
    for il in [int(x) for x in a.layers.split(",")]:
        p = out / f"L{il}.json"
        if p.is_file():
            log(f"L{il}: {p} exists, skipping")
            continue
        res = run_layer(il, Path(a.cal), Path(a.heldout), ranks, comp_ranks,
                        [int(x) for x in a.diag_ranks.split(",")], tb=a.tb, max_heldout=a.max_heldout,
                        fit_cache=Path(a.fit_cache) if a.fit_cache else None, fit_only=a.fit_only)
        if a.fit_only:
            continue
        res["host"] = host
        res["generator"] = "pigemma.h2.d1_lowrank_residual"
        res["inputs"] = {"cal": str(a.cal), "heldout": str(a.heldout), "map": MAP31, "lambda_whiten": LAM, "block": GC}
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"L{il}: wrote {p} ({res['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
