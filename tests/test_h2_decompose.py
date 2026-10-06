"""Tests for pigemma.h2.decompose (manifest files[] 'pigemma/h2/decompose.py', TRACE_FORMAT 13.3).

Unit tests run on Windows without model files: Q4_0 bytes come from gguf-py's quantiser and a
tiny dense gemma4 GGUF is written with gguf.GGUFWriter (no tests/conftest.py fixtures needed).
Assertions from the manifest test field:

* random f32 [64,256] -> gguf.quants.quantize(Q4_0) -> dequant_q4_0 == gguf.quants.dequantize
  bit-exactly;
* group_quant(4) error <= Q4_0 reference error * 1.05 on Gaussian data (it is in fact identical),
  and the bytes/elem formula (q2 0.3125, q3 0.4375, q4 0.5625);
* lowrank(r=min(dims)) reconstructs to 1e-4 (float32 factors; the default f16 factors are checked
  against the f16 rounding bound instead, since f16 alone costs ~3e-4 relative);
* decompose(res_bits=16): W_base_hat + sum_g R_g == W within 1e-5;
* bound soundness: for 1000 random x, ||R_g x_g|| <= norm2_g*||x_g|| and <= normF_g*||x_g||;
* bytes accounting = base + sum(bytes_g).

``pytest -m integration`` additionally checks the real 31B Q4_0 matrices (skipped if absent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType as QT  # noqa: E402

from pigemma import gguf_map as gm  # noqa: E402
from pigemma import paths  # noqa: E402
from pigemma.h2 import decompose as dc  # noqa: E402


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


BACKENDS = ["numpy"] + (["torch"] if _torch_ok() else [])


def _u32(a: Any) -> Any:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _rel(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)) / np.linalg.norm(np.asarray(b, np.float64)))


# ----------------------------------------------------------------------------------------- Q4_0
@pytest.mark.parametrize("shape,seed", [((64, 256), 0), ((64, 256), 1), ((3, 32), 2), ((17, 96), 3)])
def test_dequant_q4_0_bitexact_vs_gguf(shape: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(shape).astype(np.float32)
    q = gguf.quants.quantize(W, QT.Q4_0)
    ref = gguf.quants.dequantize(q, QT.Q4_0)
    for buf in (q.tobytes(), bytearray(q.tobytes()), memoryview(q.tobytes()), q):
        mine = dc.dequant_q4_0(buf, *shape)
        assert mine.shape == shape and mine.dtype == np.float32
        assert np.array_equal(_u32(mine), _u32(ref))


def test_dequant_q4_0_block_layout() -> None:
    # One block: d = 0.5; byte j = low nibble j | high nibble (15 - j).
    d = np.array([0.5], dtype=np.float16).view(np.uint8)
    qs = np.array([(j & 0xF) | ((15 - j) << 4) for j in range(16)], dtype=np.uint8)
    blk = np.concatenate([d, qs]).tobytes()
    out = dc.dequant_q4_0(blk, 1, 32)[0]
    expect = [0.5 * (j - 8) for j in range(16)] + [0.5 * ((15 - j) - 8) for j in range(16)]
    assert out.tolist() == expect


def test_dequant_q4_0_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        dc.dequant_q4_0(b"\0" * 18, 1, 48)       # n_cols not a multiple of 32
    with pytest.raises(ValueError):
        dc.dequant_q4_0(b"\0" * 17, 1, 32)       # wrong byte count


# ----------------------------------------------------------------------------------- group quant
@pytest.mark.parametrize("scale", [1.0, 0.02])
def test_group_quant_4_matches_q4_0_reference(scale: float) -> None:
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((64, 256)) * scale).astype(np.float32)
    ref = gguf.quants.dequantize(gguf.quants.quantize(W, QT.Q4_0), QT.Q4_0)
    W_hat, nbytes = dc.group_quant(W, 4)
    err_ref = np.linalg.norm(W - ref)
    err = np.linalg.norm(W - W_hat)
    assert err <= err_ref * 1.05
    assert np.array_equal(_u32(W_hat), _u32(ref))  # same rule -> identical values
    assert nbytes == q4_bytes(64, 256) == gguf.quants.quantize(W, QT.Q4_0).nbytes


def q4_bytes(n_out: int, n_in: int) -> int:
    return dc.q4_0_nbytes(n_out, n_in)


def test_group_quant_bytes_formula() -> None:
    W = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    named = {2: 0.3125, 3: 0.4375, 4: 0.5625, 8: 1.0625}
    for bits, bpe in named.items():
        _, nbytes = dc.group_quant(W, bits)
        assert nbytes / W.size == pytest.approx(bpe, abs=0)
        assert dc.bytes_per_elem(bits) == bpe == bits / 8 + 2 / 32
    # Partial blocks are stored as whole blocks: 100 columns -> 4 blocks per row.
    _, nb = dc.group_quant(W[:, :100], 4)
    assert nb == 64 * 4 * 18
    assert dc.q4_0_nbytes(5376, 21504) == 65_028_096 == dc.q4_0_nbytes(21504, 5376)


def test_group_quant_error_and_idempotence() -> None:
    W = np.random.default_rng(3).standard_normal((32, 320)).astype(np.float32)
    errs = []
    for bits in (2, 3, 4, 5, 6, 8):
        W_hat, _ = dc.group_quant(W, bits)
        assert W_hat.dtype == np.float32 and W_hat.shape == W.shape
        errs.append(_rel(W_hat, W))
        again, _ = dc.group_quant(W_hat, bits)
        assert np.array_equal(again, W_hat), f"requantising q{bits} must be lossless"
    assert all(a > b for a, b in zip(errs, errs[1:]))
    # Partial last block + zero block.
    Z = np.zeros((4, 70), np.float32)
    Z[:, :5] = 1.0
    Zh, _ = dc.group_quant(Z, 3)
    assert np.isfinite(Zh).all() and np.array_equal(Zh[:, 32:], Z[:, 32:])
    with pytest.raises(ValueError):
        dc.group_quant(W, 1)
    with pytest.raises(ValueError):
        dc.group_quant(W, 9)


# --------------------------------------------------------------------------------- base names
def test_parse_base() -> None:
    assert dc.parse_base("q2") == dc.BaseSpec("q2", 2, None)
    assert dc.parse_base("Q3") == dc.BaseSpec("q3", 3, None)
    assert dc.parse_base("lowrank256") == dc.BaseSpec("lowrank256", None, 256)
    assert dc.parse_base("q2+lowrank128") == dc.BaseSpec("q2+lowrank128", 2, 128)
    assert dc.parse_base("none") == dc.BaseSpec("none", None, None)
    for name in dc.BASE_EXAMPLES:
        assert dc.parse_base(name).name == name
    for bad in ("", "q1", "q9", "lowrank0", "lowrank", "lowrank128+q2", "q2lowrank128", "svd64"):
        with pytest.raises(ValueError, match="unknown base"):
            dc.parse_base(bad)
    assert dc.parse_base("q2+lowrank128").base_bytes(10, 64) == 10 * 2 * 10 + 2 * 128 * 74


def test_ffn_names() -> None:
    assert dc.ffn_names(20) == {"gate": "blk.20.ffn_gate.weight", "up": "blk.20.ffn_up.weight",
                                "down": "blk.20.ffn_down.weight"}
    assert dc.FFN_INPUT_KIND == {"gate": "ffn_norm", "up": "ffn_norm", "down": "ffn_geglu"}


# ------------------------------------------------------------------------------------ low rank
@pytest.mark.parametrize("backend", BACKENDS)
def test_lowrank_reconstructs(backend: str) -> None:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((64, 256)).astype(np.float32)
    # r = min(dims): exact up to rounding.
    U, S, Vt, nbytes = dc.lowrank(W, 64, backend=backend, factor_dtype=np.float32)
    assert _rel((U * S) @ Vt, W) <= 1e-4
    assert nbytes == 4 * 64 * (64 + 256)
    # A genuine rank-8 matrix with r = 8.
    A = (rng.standard_normal((64, 8)) @ rng.standard_normal((8, 256))).astype(np.float32)
    U, S, Vt, _ = dc.lowrank(A, 8, backend=backend, factor_dtype=np.float32)
    assert _rel((U * S) @ Vt, A) <= 1e-4
    assert np.all(np.diff(S) <= 1e-5 * S[0])
    # Default f16 factors: spec shapes/dtypes/bytes, error bounded by f16 rounding.
    U, S, Vt, nbytes = dc.lowrank(W, 64, backend=backend)
    assert U.dtype == np.float16 and Vt.dtype == np.float16 and S.dtype == np.float32
    assert U.shape == (64, 64) and Vt.shape == (64, 256) and S.shape == (64,)
    assert nbytes == 2 * 64 * (256 + 64)
    assert _rel((U.astype(np.float32) * S) @ Vt.astype(np.float32), W) <= 2e-3
    with pytest.raises(ValueError):
        dc.lowrank(W, 65, backend=backend)
    with pytest.raises(ValueError):
        dc.lowrank(W, 0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lowrank_truncation_and_seed(backend: str) -> None:
    rng = np.random.default_rng(2)
    W = rng.standard_normal((96, 160)).astype(np.float32)
    a = dc.lowrank(W, 16, backend=backend, seed=5)
    b = dc.lowrank(W, 16, backend=backend, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a[:3], b[:3]))
    s_exact = np.linalg.svd(W.astype(np.float64), compute_uv=False)[:16]
    # Gaussian matrices have a flat spectrum: the randomized SVD is close but not exact.
    assert np.allclose(a[1], s_exact, rtol=0.05)
    info: dict[str, Any] = {}
    dc.lowrank(W, 16, backend=backend, info=info)
    assert info["backend"] == backend and info["q"] == 16 + 32


# ---------------------------------------------------------------------------------- decompose
BASES = ["q2", "q3", "lowrank16", "q2+lowrank8", "none"]


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("group", [128, 100])
def test_decompose_res16_is_exact(base: str, group: int) -> None:
    W = np.random.default_rng(11).standard_normal((64, 256)).astype(np.float32)
    d = dc.decompose(W, base, group_cols=group, res_bits=16, lowrank_backend="numpy")
    total = d.W_base_hat.astype(np.float64).copy()
    for g in d.groups:
        assert g.R_hat.dtype == np.float32 and g.R_hat.shape == (64, g.width)
        total[:, g.cols] += g.R_hat
    assert np.max(np.abs(total - W)) <= 1e-5
    assert np.max(np.abs(d.reconstruct() - W)) <= 1e-5
    assert d.W_base_hat.dtype == np.float32
    assert np.array_equal(d.W_base_hat.astype(np.float16).astype(np.float32), d.W_base_hat)
    assert d.meta["recon_rel_err"] <= 1e-6


def test_decompose_groups_padding_and_cols() -> None:
    W = np.random.default_rng(4).standard_normal((48, 256)).astype(np.float32)
    d = dc.decompose(W, "q2", group_cols=100, res_bits=4)
    assert [g.width for g in d.groups] == [100, 100, 56]
    assert d.cols.tolist() == [[0, 100], [100, 200], [200, 256]]
    assert d.R_stack is not None and d.R_stack.shape == (3, 48, 100) and d.R_stack.dtype == np.float16
    assert not np.any(d.R_stack[2, :, 56:])          # short last group zero-padded on the right
    assert d.n_groups == 3 and d.n_out == 48 and d.n_in == 256
    assert dc.group_slices(5376, 128)[-1] == slice(5248, 5376) and len(dc.group_slices(21504, 128)) == 168
    with pytest.raises(ValueError):
        dc.decompose(W, "q2", res_bits=9)
    with pytest.raises(ValueError):
        dc.decompose(W, "lowrank64")                 # rank > min(n_out, n_in)


@pytest.mark.parametrize("res_bits", [4, 8, 16])
@pytest.mark.parametrize("base", ["q2", "lowrank16"])
def test_bound_soundness(base: str, res_bits: int) -> None:
    rng = np.random.default_rng(21)
    W = rng.standard_normal((64, 256)).astype(np.float32)
    d = dc.decompose(W, base, group_cols=64, res_bits=res_bits, lowrank_backend="numpy")
    X = rng.standard_normal((1000, 256)) * rng.uniform(0.01, 10.0, size=(1000, 1))
    for g in d.groups:
        R = g.R_hat.astype(np.float64)
        xg = X[:, g.cols]
        lhs = np.linalg.norm(xg @ R.T, axis=1)
        xn = np.linalg.norm(xg, axis=1)
        assert np.all(lhs <= g.norm2_g * xn), "spectral bound violated"
        assert np.all(lhs <= g.normF_g * xn), "Frobenius bound violated"
        s = np.linalg.svd(R, compute_uv=False)
        assert s[0] <= g.norm2_g <= s[0] * (1 + 2 * dc.NORM2_RTOL)
        assert g.normF_g == pytest.approx(np.linalg.norm(R), rel=1e-12)
        assert g.norm2_g <= g.normF_g * (1 + 2 * dc.NORM2_RTOL)
        # Adversarial input: the top right singular vector.
        v = np.linalg.svd(R)[2][0]
        assert np.linalg.norm(R @ v) <= g.norm2_g * np.linalg.norm(v)
    # The batched helpers give the same numbers.
    C = d.contributions(X[:32])
    N = d.input_norms(X[:32])
    assert C.shape == (32, d.n_groups, 64) and N.shape == (32, d.n_groups)
    assert np.all(np.linalg.norm(C.astype(np.float64), axis=2) <= d.norm2[None, :] * N * (1 + 1e-5))


def test_contributions_sum_to_reconstruction() -> None:
    rng = np.random.default_rng(5)
    W = rng.standard_normal((40, 160)).astype(np.float32)
    d = dc.decompose(W, "q3", group_cols=48, res_bits=4)
    X = rng.standard_normal((7, 160)).astype(np.float32)
    y = X @ d.W_base_hat.T + d.contributions(X).sum(axis=1)
    assert np.allclose(y, X @ d.reconstruct().T, atol=1e-4)
    assert np.allclose(d.reconstruct(groups=[]), d.W_base_hat)


@pytest.mark.parametrize("res_bits,res_bpe", [(4, 0.5625), (8, 1.0625), (16, 2.0)])
@pytest.mark.parametrize("base,base_bytes", [("q2", 64 * 256 * 0.3125), ("q3", 64 * 256 * 0.4375),
                                             ("lowrank16", 2 * 16 * (256 + 64)),
                                             ("q2+lowrank8", 64 * 256 * 0.3125 + 2 * 8 * (256 + 64)),
                                             ("none", 0)])
def test_bytes_accounting(base: str, base_bytes: float, res_bits: int, res_bpe: float) -> None:
    W = np.random.default_rng(6).standard_normal((64, 256)).astype(np.float32)
    d = dc.decompose(W, base, group_cols=128, res_bits=res_bits, lowrank_backend="numpy")
    assert d.base_bytes == int(base_bytes)
    assert d.total_residual_bytes == sum(g.bytes_g for g in d.groups) == int(d.bytes_g.sum())
    assert d.total_residual_bytes == int(res_bpe * W.size)
    assert d.total_bytes == d.base_bytes + d.total_residual_bytes
    assert all(g.bytes_g == dc.residual_bytes(64, g.width, res_bits) for g in d.groups)
    assert d.meta["q4_0_bytes"] == dc.q4_0_nbytes(64, 256)
    assert dc.parse_base(base).base_bytes(64, 256) == d.base_bytes


def test_q2_plus_lowrank_not_worse_than_q2() -> None:
    rng = np.random.default_rng(8)
    W = (rng.standard_normal((96, 128)) + rng.standard_normal((96, 4)) @ rng.standard_normal((4, 128))).astype(np.float32)
    e_q2 = dc.decompose(W, "q2", res_bits=16).meta["base_rel_err"]
    e_mix = dc.decompose(W, "q2+lowrank16", res_bits=16, lowrank_backend="numpy").meta["base_rel_err"]
    assert e_mix <= e_q2


def test_res4_reconstruction_floor_reported() -> None:
    W = np.random.default_rng(9).standard_normal((64, 256)).astype(np.float32)
    d4 = dc.decompose(W, "q2", res_bits=4)
    d8 = dc.decompose(W, "q2", res_bits=8)
    assert 0 < d8.meta["recon_rel_err"] < d4.meta["recon_rel_err"] < 0.1
    assert d4.meta["recon_rel_err"] == pytest.approx(_rel(d4.reconstruct(), W), rel=1e-3)


# ----------------------------------------------------------------------------------------- npz
def test_npz_schema_and_roundtrip(tmp_path: Path) -> None:
    W = np.random.default_rng(10).standard_normal((48, 224)).astype(np.float32)
    d = dc.decompose(W, "q2+lowrank8", group_cols=96, res_bits=4, lowrank_backend="numpy")
    p = dc.decomp_path(tmp_path / "decomp", 20, "down", "q2+lowrank8")
    assert p.name == "L20.down.q2+lowrank8.npz"
    out = dc.save_npz(d, p, layer=20, mat="down", tensor="blk.20.ffn_down.weight", gguf="x.gguf",
                      orig_type="Q4_0", orig_bytes=dc.q4_0_nbytes(48, 224))
    assert out == p and p.is_file() and not list(p.parent.glob("*.tmp*"))
    with np.load(p, allow_pickle=False) as z:
        assert z["W_base_hat"].dtype == np.float16 and z["W_base_hat"].shape == (48, 224)
        assert z["R_hat"].dtype == np.float16 and z["R_hat"].shape == (3, 48, 96)
        assert z["cols"].dtype == np.int64 and z["cols"].tolist() == [[0, 96], [96, 192], [192, 224]]
        assert z["norm2"].dtype == np.float64 and z["normF"].dtype == np.float64
        assert z["bytes_g"].dtype == np.int64 and z["bytes_g"].shape == (3,)
        for k in ("base_bytes", "q4_0_bytes", "layer", "group_cols", "res_bits", "n_out", "n_in"):
            assert z[k].dtype == np.int64 and z[k].ndim == 0, k
        assert z["mat"].dtype.kind == "U" and str(z["mat"]) == "down"
        assert str(z["base"]) == "q2+lowrank8"
        assert int(z["q4_0_bytes"]) == dc.q4_0_nbytes(48, 224)
        assert z["lowrank_S"].shape == (8,)
        meta = json.loads(str(z["meta_json"]))
        assert meta["base"] == "q2+lowrank8" and meta["layer"] == 20 and meta["tensor"] == "blk.20.ffn_down.weight"
    z = dc.load_npz(p)
    assert isinstance(z["layer"], int) and z["mat"] == "down" and z["res_bits"] == 4
    d2 = dc.Decomp.from_npz(p)
    assert np.array_equal(d2.W_base_hat, d.W_base_hat)
    assert np.array_equal(d2.reconstruct(), d.reconstruct())
    assert d2.base_bytes == d.base_bytes and d2.total_residual_bytes == d.total_residual_bytes
    assert np.array_equal(d2.norm2, d.norm2) and np.array_equal(d2.normF, d.normF)


def test_npz_res16_stores_exact_float32(tmp_path: Path) -> None:
    W = np.random.default_rng(12).standard_normal((32, 128)).astype(np.float32)
    d = dc.decompose(W, "q2", group_cols=64, res_bits=16)
    p = dc.save_npz(d, tmp_path / "L0.gate.q2.npz", layer=0, mat="gate")
    d2 = dc.Decomp.from_npz(p)
    assert d2.stacked().dtype == np.float32
    assert np.max(np.abs(d2.reconstruct() - W)) <= 1e-5


# --------------------------------------------------------------------------- tiny GGUF + CLI
N_EMBD, N_FF = 64, 96


def _write_tiny_dense_gguf(path: Path, seed: int = 0) -> dict[str, Any]:
    """2-layer dense gemma4 GGUF with Q4_0 FFN matrices; returns {name: Q4_0 byte array}."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "gemma4")
    a = "gemma4"
    for key, val in ((f"{a}.block_count", 2), (f"{a}.context_length", 128), (f"{a}.embedding_length", N_EMBD),
                     (f"{a}.feed_forward_length", N_FF), (f"{a}.attention.head_count", 2),
                     (f"{a}.attention.key_length", 32)):
        w.add_uint32(key, val)
    w.add_array(f"{a}.attention.head_count_kv", [2, 2])
    q4: dict[str, Any] = {}
    w.add_tensor("token_embd.weight", rng.standard_normal((16, N_EMBD)).astype(np.float16))
    w.add_tensor("output_norm.weight", rng.standard_normal(N_EMBD).astype(np.float32))
    for il in range(2):
        for name, ne in ((f"blk.{il}.ffn_gate.weight", [N_EMBD, N_FF]), (f"blk.{il}.ffn_up.weight", [N_EMBD, N_FF]),
                         (f"blk.{il}.ffn_down.weight", [N_FF, N_EMBD]), (f"blk.{il}.attn_q.weight", [N_EMBD, 64])):
            x = (rng.standard_normal(tuple(reversed(ne))) * 0.02).astype(np.float32)
            q = gguf.quants.quantize(x, QT.Q4_0)
            w.add_tensor(name, q, raw_dtype=QT.Q4_0)
            q4[name] = q
        w.add_tensor(f"blk.{il}.ffn_norm.weight", rng.standard_normal(N_EMBD).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return q4


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    d = tmp_path_factory.mktemp("h2_decompose")
    gp = d / "tiny-dense.gguf"
    q4 = _write_tiny_dense_gguf(gp)
    gmap = gm.build_map(gp, with_sha=False)
    mp = d / "map.json"
    gmap.save(mp)
    return {"dir": d, "gguf": gp, "map": mp, "q4": q4}


def test_load_matrix_orientation_and_sources(tiny: dict[str, Any]) -> None:
    for name, (n_out, n_in) in (("blk.1.ffn_gate.weight", (N_FF, N_EMBD)), ("blk.0.ffn_down.weight", (N_EMBD, N_FF))):
        ref = gguf.quants.dequantize(tiny["q4"][name], QT.Q4_0)          # [ne1, ne0] = [n_out, n_in]
        for src in (tiny["map"], str(tiny["map"]), gm.GgufMap.load(tiny["map"]),
                    gm.GgufMap.load(tiny["map"]).to_dict(), tiny["gguf"], str(tiny["gguf"])):
            W = dc.load_matrix(src, name)
            assert W.shape == (n_out, n_in) and W.dtype == np.float32 and W.flags.c_contiguous
            assert np.array_equal(_u32(W), _u32(ref))
        info = dc.tensor_info(tiny["map"], name)
        assert (info.n_out, info.n_in) == (n_out, n_in) and info.type == "Q4_0"
        assert info.n_bytes == dc.q4_0_nbytes(n_out, n_in)
        # Re-quantising a Q4_0 matrix at 4 bits is lossless (checks the block orientation).
        assert np.array_equal(dc.group_quant(W, 4)[0], W)
    with pytest.raises(KeyError):
        dc.load_matrix(tiny["map"], "blk.9.ffn_gate.weight")
    with pytest.raises(KeyError):
        dc.load_matrix(tiny["gguf"], "blk.9.ffn_gate.weight")
    with pytest.raises(FileNotFoundError):
        dc.load_matrix(tiny["dir"] / "nope.json", "blk.0.ffn_gate.weight")
    with pytest.raises(ValueError):
        dc.load_matrix(tiny["map"], "output_norm.weight")              # 1-D: not a matrix


def test_resolve_gguf_size_mismatch(tiny: dict[str, Any], tmp_path: Path) -> None:
    doc = json.loads(Path(tiny["map"]).read_text(encoding="utf-8"))
    doc["file_size"] += 1
    with pytest.raises(ValueError, match="file size"):
        dc.load_matrix(doc, "blk.0.ffn_gate.weight")
    doc["file_size"] -= 1
    doc["gguf"] = str(tmp_path / "moved-away.gguf")
    doc["gguf_basename"] = "moved-away.gguf"
    with pytest.raises(FileNotFoundError):
        dc.load_matrix(doc, "blk.0.ffn_gate.weight")
    assert dc.load_matrix(doc, "blk.0.ffn_gate.weight", gguf=tiny["gguf"]).shape == (N_FF, N_EMBD)


def test_cli_end_to_end(tiny: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "decomp"
    js = tmp_path / "run.json"
    argv = ["--map", str(tiny["map"]), "--layers", "0,1", "--mats", "gate,down", "--bases", "q2,lowrank8",
            "--group", "32", "--res-bits", "4", "--out", str(out), "--json", str(js), "--lowrank-backend", "numpy"]
    assert dc.main(argv) == 0
    cap = capsys.readouterr()
    files = sorted(p.name for p in out.glob("*.npz"))
    assert files == sorted(f"L{il}.{m}.{b}.npz" for il in (0, 1) for m in ("gate", "down") for b in ("q2", "lowrank8"))
    assert "Bytes accounting vs Q4_0" in cap.out and "L1.down.lowrank8.npz" in cap.out
    assert "ETA" in cap.err
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert len(idx["entries"]) == 8
    run = json.loads(js.read_text(encoding="utf-8"))
    assert len(run["entries"]) == 8 and all(not e["cached"] for e in run["entries"])
    z = dc.load_npz(out / "L0.down.q2.npz")
    assert (z["n_out"], z["n_in"], z["layer"], z["mat"], z["base"]) == (N_EMBD, N_FF, 0, "down", "q2")
    assert z["tensor"] == "blk.0.ffn_down.weight" and z["gguf"] == "tiny-dense.gguf" and z["orig_type"] == "Q4_0"
    assert z["R_hat"].shape == (3, N_EMBD, 32) and z["q4_0_bytes"] == dc.q4_0_nbytes(N_EMBD, N_FF)
    e = next(e for e in run["entries"] if e["file"] == "L0.down.q2.npz")
    assert e["base_bytes"] + e["total_residual_bytes"] == z["base_bytes"] + int(z["bytes_g"].sum())
    assert e["base_bytes_per_elem"] == pytest.approx(0.3125) and e["residual_bytes_per_elem"] == pytest.approx(0.5625)

    # Second run reuses everything; --force recomputes.
    mtime = (out / "L0.down.q2.npz").stat().st_mtime_ns
    assert dc.main(argv) == 0
    cap = capsys.readouterr()
    assert cap.err.count("reusing") == 8 and (out / "L0.down.q2.npz").stat().st_mtime_ns == mtime
    assert dc.main(argv[:-2] + ["--lowrank-backend", "numpy", "--force", "--mats", "down", "--bases", "q2",
                                "--layers", "0"]) == 0
    capsys.readouterr()
    # Different parameters -> not reused.
    assert dc.main(["--map", str(tiny["map"]), "--layers", "0", "--mats", "gate", "--bases", "q2", "--group", "64",
                    "--out", str(out)]) == 0
    assert "reusing" not in capsys.readouterr().err

    # Dry run writes nothing.
    dry = tmp_path / "dry"
    assert dc.main(["--map", str(tiny["map"]), "--layers", "0", "--bases", "q3", "--out", str(dry), "--dry-run"]) == 0
    cap = capsys.readouterr()
    assert "DRY RUN" in cap.out and not dry.exists()

    # JSON to stdout.
    assert dc.main(["--map", str(tiny["map"]), "--layers", "1", "--mats", "up", "--bases", "none",
                    "--out", str(out), "--json", "-"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["entries"][0]["file"] == "L1.up.none.npz"

    # Also works from the GGUF directly (no map).
    assert dc.main(["--gguf", str(tiny["gguf"]), "--layers", "1", "--mats", "up", "--bases", "q2",
                    "--out", str(tmp_path / "from_gguf")]) == 0
    capsys.readouterr()
    assert (tmp_path / "from_gguf" / "L1.up.q2.npz").is_file()


@pytest.mark.parametrize("extra,code", [
    (["--bases", "q9"], paths.EXIT_USAGE),
    (["--layers", "5"], paths.EXIT_USAGE),
    (["--mats", "gate,ffn"], paths.EXIT_USAGE),
    (["--layers", "x"], paths.EXIT_USAGE),
    (["--bases", "lowrank999"], paths.EXIT_USAGE),
    (["--min-free-gb", "1e12"], paths.EXIT_REFUSED),
])
def test_cli_exit_codes(tiny: dict[str, Any], tmp_path: Path, extra: list[str], code: int,
                        capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--map", str(tiny["map"]), "--layers", "0", "--mats", "gate", "--bases", "q2",
            "--out", str(tmp_path / "o")] + extra
    assert dc.main(argv) == code
    err = capsys.readouterr().err
    if code == paths.EXIT_REFUSED:
        assert "REFUSED" in err
    assert not list((tmp_path / "o").glob("*.npz")) if (tmp_path / "o").exists() else True


def test_cli_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert dc.main(["--map", str(tmp_path / "none.json"), "--out", str(tmp_path / "o")]) == paths.EXIT_MISSING
    assert dc.main(["--gguf", str(tmp_path / "none.gguf"), "--out", str(tmp_path / "o")]) == paths.EXIT_MISSING
    assert dc.main(["--out", str(tmp_path / "o")]) == paths.EXIT_USAGE
    with pytest.raises(SystemExit) as ex:
        dc.main(["--res-bits", "7"])
    assert ex.value.code == 2
    capsys.readouterr()


def test_module_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ex:
        dc.main(["--help"])
    assert ex.value.code == 0
    assert "--res-bits" in capsys.readouterr().out


# --------------------------------------------------------------------------------- integration
MAP31 = paths.TRACES_DIR / "h2" / "map31.json"


@pytest.mark.integration
def test_real_31b_matrices() -> None:
    if not MAP31.is_file():
        pytest.skip("traces/h2/map31.json missing")
    gmap = gm.GgufMap.load(MAP31)
    try:
        dc.resolve_gguf(gmap)
    except FileNotFoundError:
        pytest.skip("31B GGUF not present")
    for mat, shape in (("down", (5376, 21504)), ("gate", (21504, 5376))):
        name = dc.ffn_names(20)[mat]
        info = dc.tensor_info(gmap, name)
        assert info.type == "Q4_0" and info.n_bytes == 65_028_096 == dc.q4_0_nbytes(*shape)
        W = dc.load_matrix(gmap, name)
        assert W.shape == shape
        row_bytes = info.n_in // 32 * 18
        mm = np.memmap(info.path, dtype=np.uint8, mode="r", offset=info.offset, shape=(info.n_bytes,))
        ref = gguf.quants.dequantize(np.array(mm[: 64 * row_bytes]).reshape(64, row_bytes), QT.Q4_0)
        del mm
        assert np.array_equal(_u32(W[:64]), _u32(ref))
        assert np.array_equal(dc.group_quant(W, 4)[0], W)       # lossless: orientation is right
        d = dc.decompose(W[:256], "q2", res_bits=16)
        assert np.max(np.abs(d.reconstruct() - W[:256])) <= 1e-6
        assert 0.2 < d.meta["base_rel_err"] < 0.6
        del W, d
