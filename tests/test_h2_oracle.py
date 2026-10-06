"""Tests for pigemma.h2.oracle (manifest files[] 'pigemma/h2/oracle.py', experiment E7).

Unit tests only: synthetic [64, 256] matrices and the conftest ``synth_acts_dir`` (an act dump over
``tiny_dense_gguf`` whose ffn_out = W_down @ ffn_geglu). No model files, no Docker. Assertions from
the manifest test field:

* W with a planted residual concentrated in ~10% of the column groups -> oracle fraction <= 0.15
  at eps 0.01, and the certified-skip set never contains a planted group;
* the oracle fraction is non-increasing in eps (per token and in the mean);
* eps = inf -> 0 groups read; W_base == W -> fraction 0;
* soundness_violations == 0 over 200 random cases (re-verified independently from the group sets);
* Spearman in [-1, 1] (and equal to a reference rank correlation);
* the sanity check passes on synthetic ffn_out = W_down @ geglu (and aborts on a scaled ffn_out);
* the markdown renders and contains the verdict string.

Plus: CLI end to end (data_formats 'h2 oracle.json' keys, phases, envelopes valid for
pigemma.results, --render-from), exit codes, the verdict rule table, the iso-byte helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from pigemma import results as rs  # noqa: E402
from pigemma.h2 import decompose as dc  # noqa: E402
from pigemma.h2 import oracle as orc  # noqa: E402

EPS3 = (0.01, 0.03, 0.10)


# ------------------------------------------------------------------------------------------ helpers
def _norm2(R: Any) -> float:
    R64 = np.asarray(R, dtype=np.float64)
    if R64.size == 0 or not np.any(R64):
        return 0.0
    return float(np.linalg.svd(R64, compute_uv=False)[0]) * (1.0 + dc.NORM2_RTOL)


def _decomp(W_base: Any, R_groups: list[Any], group_cols: int, *, dtype: Any = np.float32,
            bytes_per_group: int | None = None) -> orc.DecompArrays:
    """DecompArrays from a base and per-group residual blocks (exact stored values)."""
    W_base = np.asarray(W_base, dtype=np.float32)
    n_out, n_in = W_base.shape
    cols = [[a, min(a + group_cols, n_in)] for a in range(0, n_in, group_cols)]
    assert len(cols) == len(R_groups)
    g_max = cols[0][1] - cols[0][0]
    R_stack = np.zeros((len(cols), n_out, g_max), dtype=dtype)
    for i, ((a, b), Rg) in enumerate(zip(cols, R_groups)):
        R_stack[i, :, : b - a] = np.asarray(Rg, dtype=dtype)
    norm2 = np.array([_norm2(R_stack[i, :, : b - a]) for i, (a, b) in enumerate(cols)])
    if bytes_per_group is None:
        bytes_g = np.array([dc.residual_bytes(n_out, b - a, 4) for a, b in cols], dtype=np.int64)
    else:
        bytes_g = np.full(len(cols), int(bytes_per_group), dtype=np.int64)
    return orc.DecompArrays(W_base, R_stack, np.array(cols, dtype=np.int64), norm2, bytes_g,
                            base_bytes=dc.quant_bytes(n_out, n_in, 2), meta={"block": 32})


def _planted(seed: int = 0, *, n_out: int = 64, n_in: int = 256, g: int = 8, planted: tuple[int, ...] = (3, 17, 28),
             scale: float = 4.0) -> tuple[orc.DecompArrays, Any, Any]:
    """W = W_base + R with R non-zero only in the planted groups (3 of 32 = 9.4%)."""
    rng = np.random.default_rng(seed)
    W_base = rng.standard_normal((n_out, n_in)).astype(np.float32)
    G = n_in // g
    R = [np.zeros((n_out, g), np.float32) for _ in range(G)]
    for p in planted:
        R[p] = (scale * rng.standard_normal((n_out, g))).astype(np.float32)
    D = _decomp(W_base, R, g, bytes_per_group=100)
    return D, D.reconstruct(), rng


# ---------------------------------------------------------------------- planted residual / basics
def test_planted_residual_oracle_and_certificate() -> None:
    planted = (3, 17, 28)
    D, W, rng = _planted(planted=planted)
    X = rng.standard_normal((96, D.n_in)).astype(np.float32)
    ts = orc.analyze_case(D, W, X, eps=EPS3, batch=16, details=True)
    assert ts.valid.all()
    agg = orc.aggregate_case(ts, "all")
    frac01 = agg["oracle"]["0.01"]["frac_bytes_mean"]
    assert frac01 is not None and frac01 <= 0.15
    assert agg["oracle"]["0.01"]["unreachable_frac"] == 0.0
    non = [g for g in range(D.n_groups) if g not in planted]
    # the oracle reads planted groups only; the certificate never skips one
    assert not ts.oracle_mask[:, :, non].any()
    assert not ts.cert_mask[:, :, list(planted)].any()
    # ... and it does certify every zero-residual group (b_g = 0)
    assert ts.cert_mask[:, :, non].all()
    for k in ("0.01", "0.03", "0.10"):
        b = agg["bound"][k]
        assert b["soundness_violations"] == 0
        assert b["certified_frac_bytes_mean"] == pytest.approx(len(non) / D.n_groups)
        # exact decomposition: same-basis and literal efficiency coincide and are 1 here
        assert b["certification_efficiency"] == pytest.approx(1.0)
        assert b["certification_efficiency_vs_floor_oracle"] == pytest.approx(1.0)
    assert agg["bound_violations"] == 0
    assert agg["floor_err_mean"] < 1e-6


def test_fraction_non_increasing_in_eps() -> None:
    rng = np.random.default_rng(1)
    W, _ = dc.group_quant(rng.standard_normal((64, 256)).astype(np.float32), 4)
    X = rng.standard_normal((80, 256)).astype(np.float32)
    X[:, rng.choice(256, 6, replace=False)] *= 15.0
    eps = (0.005, 0.01, 0.03, 0.10, 0.3)
    for res_bits in (4, 16):
        d = dc.decompose(W, "q2", 16, res_bits)
        ts = orc.analyze_case(d, W, X, eps=eps, batch=7)
        k = ts.k_oracle
        assert (np.diff(k, axis=1) <= 0).all(), "groups read must not grow with eps"
        assert (np.diff(ts.reach.astype(int), axis=1) >= 0).all()
        assert (np.diff(ts.k_free, axis=1) <= 0).all()
        assert (np.diff(ts.cert_skip_bytes, axis=1) >= 0).all(), "certified skip must not shrink with eps"
        agg = orc.aggregate_case(ts, "all")
        cons = [agg["oracle"][orc.eps_key(e)]["frac_bytes_mean_conservative"] for e in eps]
        assert all(a >= b - 1e-12 for a, b in zip(cons, cons[1:]))
        skip = [agg["bound"][orc.eps_key(e)]["certified_frac_bytes_mean"] for e in eps]
        assert all(a <= b + 1e-12 for a, b in zip(skip, skip[1:]))


def test_eps_inf_reads_zero_groups() -> None:
    D, W, rng = _planted(seed=2)
    X = rng.standard_normal((20, D.n_in)).astype(np.float32)
    eps = orc.parse_eps("inf")
    assert eps == (math.inf,) and orc.eps_key(math.inf) == "inf"
    ts = orc.analyze_case(D, W, X, eps=eps)
    assert (ts.k_oracle == 0).all() and ts.reach.all()
    agg = orc.aggregate_case(ts, "all")
    assert agg["oracle"]["inf"]["frac_bytes_mean"] == 0.0
    assert agg["oracle"]["inf"]["k_groups_mean"] == 0.0
    assert agg["bound"]["inf"]["soundness_violations"] == 0


def test_base_equal_to_W_needs_no_residual() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((64, 256)).astype(np.float32)
    D = _decomp(W, [np.zeros((64, 32), np.float32) for _ in range(8)], 32)
    X = rng.standard_normal((40, 256)).astype(np.float32)
    ts = orc.analyze_case(D, W, X, eps=EPS3)
    assert np.allclose(ts.err_base, 0.0) and (ts.k_oracle == 0).all()
    agg = orc.aggregate_case(ts, "all")
    for k in ("0.01", "0.03", "0.10"):
        assert agg["oracle"][k]["frac_bytes_mean"] == 0.0
        assert agg["oracle"][k]["k_groups_mean"] == 0.0
        assert agg["bound"][k]["certified_frac_bytes_mean"] == 1.0   # b_g = 0 everywhere
        assert agg["bound"][k]["certification_efficiency"] == pytest.approx(1.0)
    assert agg["err_base_mean"] == pytest.approx(0.0, abs=1e-12)
    # the same through decompose: a 'none' base with W itself as the base is not expressible, so
    # check the q2 decomposition of an all-zero matrix instead (base 0, residual 0)
    z = dc.decompose(np.zeros((64, 256), np.float32), "q2", 32, 4)
    Xz = rng.standard_normal((5, 256)).astype(np.float32)
    tz = orc.analyze_case(z, np.zeros((64, 256), np.float32), Xz, eps=EPS3)
    assert not tz.valid.any()                     # y_ref == 0: dropped and counted, never divided
    assert orc.aggregate_case(tz, "all") is None


# ------------------------------------------------------------------------------------- soundness
def test_soundness_200_random_cases() -> None:
    rng = np.random.default_rng(12345)
    n_viol = n_checked = 0
    for case in range(200):
        n_out = int(rng.integers(3, 40))
        n_in = int(rng.integers(4, 120))
        g = int(rng.integers(1, 33))
        cols = [(a, min(a + g, n_in)) for a in range(0, n_in, g)]
        W_base = rng.standard_normal((n_out, n_in)).astype(np.float32)
        R = []
        for a, b in cols:
            if rng.random() < 0.15:
                R.append(np.zeros((n_out, b - a), np.float32))
            else:
                R.append((10 ** rng.uniform(-3, 0.5) * rng.standard_normal((n_out, b - a))).astype(np.float32))
        dt = np.float16 if rng.random() < 0.5 else np.float32
        D = _decomp(W_base, R, g, dtype=dt)
        W_ref = D.reconstruct()
        if rng.random() < 0.5:                    # a residual quantisation floor
            W_ref = W_ref + (10 ** rng.uniform(-4, -1) * rng.standard_normal(W_ref.shape)).astype(np.float32)
        T = int(rng.integers(1, 40))
        X = rng.standard_normal((T, n_in)).astype(np.float32)
        if rng.random() < 0.4:
            X[:, rng.integers(0, n_in)] *= float(rng.uniform(5, 50))
        if T > 2 and rng.random() < 0.2:
            X[0] = 0.0                            # y_ref == 0 -> invalid token
        eps = tuple(sorted(set(float(e) for e in 10 ** rng.uniform(-4, -0.3, size=3))))
        if rng.random() < 0.2:
            eps = eps + (math.inf,)
        ts = orc.analyze_case(D, W_ref, X, eps=eps, batch=int(rng.integers(1, 17)), details=True)
        assert ts.bound_violations.sum() == 0, f"case {case}: per-group bound violated"
        n_viol += int(ts.cert_violation.sum())
        # independent re-verification of the certificate from the group sets
        R64 = np.asarray(D.R_stack, dtype=np.float64)
        y_ref = X.astype(np.float64) @ W_ref.astype(np.float64).T
        for t in np.flatnonzero(ts.valid):
            x = X[t].astype(np.float64)
            yn = float(np.linalg.norm(y_ref[t]))
            Cg = np.stack([R64[i, :, : b - a] @ x[a:b] for i, (a, b) in enumerate(cols)])
            bg = np.array([D.norm2[i] * np.linalg.norm(x[a:b]) for i, (a, b) in enumerate(cols)])
            for ie, e in enumerate(eps):
                P = ts.cert_mask[t, ie]
                A = Cg[P].sum(axis=0) if P.any() else np.zeros(n_out)
                assert np.linalg.norm(A) <= min(e, 1e300) * yn * (1 + 1e-9) + 1e-300
                if math.isfinite(e):
                    r = e / (1 + e)
                    assert bg[P].sum() <= r * np.linalg.norm(y_ref[t] - A) * (1 + 1e-9) + 1e-12
                n_checked += 1
        sp = ts.spearman[np.isfinite(ts.spearman)]
        assert ((sp >= -1.0) & (sp <= 1.0)).all()
        for ph in orc.PHASES:
            agg = orc.aggregate_case(ts, ph)
            if agg is not None:
                assert sum(b["soundness_violations"] for b in agg["bound"].values()) == 0
    assert n_viol == 0
    assert n_checked > 2000


def test_nonfinite_rows_are_dropped() -> None:
    D, W, rng = _planted(seed=4)
    X = rng.standard_normal((10, D.n_in)).astype(np.float32)
    X[2, 5] = np.inf
    X[7, 0] = np.nan
    ts = orc.analyze_case(D, W, X.astype(np.float16), eps=EPS3)
    assert ts.finite.sum() == 8 and ts.valid.sum() == 8
    agg = orc.aggregate_case(ts, "all")
    assert agg["n_tokens"] == 8 and all(math.isfinite(v) for v in (agg["err_base_mean"], agg["floor_err_mean"]))


# -------------------------------------------------------------------------------------- spearman
def _ref_spearman(a: Any, b: Any) -> float:
    def ranks(x: Any) -> Any:
        x = np.asarray(x, float)
        r = np.empty(len(x))
        for i, v in enumerate(x):
            r[i] = (x < v).sum() + ((x == v).sum() + 1) / 2.0
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def test_spearman_range_and_reference() -> None:
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(2, 60))
        a = rng.integers(0, 6, n).astype(float) if rng.random() < 0.5 else rng.standard_normal(n)
        b = rng.integers(0, 6, n).astype(float) if rng.random() < 0.5 else rng.standard_normal(n)
        s = orc.spearman(a, b)
        if np.all(a == a[0]) or np.all(b == b[0]):
            assert s is None
            continue
        assert -1.0 <= s <= 1.0
        assert s == pytest.approx(_ref_spearman(a, b), abs=1e-12)
    x = np.arange(10.0)
    assert orc.spearman(x, x ** 3) == pytest.approx(1.0)
    assert orc.spearman(x, -x) == pytest.approx(-1.0)
    assert orc.spearman([1.0], [2.0]) is None
    assert list(orc.rankdata([3, 1, 3, 2])) == [3.5, 1.0, 3.5, 2.0]


# ------------------------------------------------------------------------------- curve / iso-byte
def test_curve_endpoints_and_iso_entries() -> None:
    rng = np.random.default_rng(6)
    W, _ = dc.group_quant(rng.standard_normal((64, 256)).astype(np.float32), 4)
    d = dc.decompose(W, "q2", 32, 4)
    X = rng.standard_normal((48, 256)).astype(np.float32)
    phase = (np.arange(48) >= 36).astype(np.uint8)
    ts = orc.analyze_case(d, W, X, eps=EPS3, phase=phase)
    uni = {b: orc.uniform_token_errors(W, X, b) for b in sorted(orc.needed_uniform_bits(ts))}
    agg = orc.aggregate_case(ts, "all", uniform=uni)
    curve = agg["curve"]
    assert len(curve) == 21 and curve[0][0] == 0.0 and curve[-1][0] == 1.0
    assert curve[0][1] == pytest.approx(agg["err_base_mean"])
    assert curve[-1][1] == pytest.approx(agg["floor_err_mean"])
    labels = [e["label"] for e in agg["iso_byte"]]
    assert labels == ["q2", "q3", "oracle@0.03"]
    for e in agg["iso_byte"]:
        assert set(e) >= {"bytes_per_elem", "uniform_err", "base_plus_oracle_err", "base_plus_certified_err"}
    q2 = agg["iso_byte"][0]
    assert q2["bytes_per_elem"] == pytest.approx(0.3125)
    # at the q2 budget a q2 base has no residual budget: base+oracle == the q2 base error
    assert q2["base_plus_oracle_err"] == pytest.approx(agg["err_base_mean"])
    # phases partition the valid tokens
    n_p = orc.aggregate_case(ts, "prefill")["n_tokens"]
    n_g = orc.aggregate_case(ts, "gen")["n_tokens"]
    assert (n_p, n_g, agg["n_tokens"]) == (36, 12, 48)


def test_uniform_quant_control_reproduces_q4_0() -> None:
    rng = np.random.default_rng(7)
    W, _ = dc.group_quant(rng.standard_normal((64, 256)).astype(np.float32), 4)   # a Q4_0 matrix
    X = rng.standard_normal((16, 256)).astype(np.float32)
    assert np.nanmax(orc.uniform_token_errors(W, X, 4)) < 1e-6
    e2 = np.nanmean(orc.uniform_token_errors(W, X, 2))
    e3 = np.nanmean(orc.uniform_token_errors(W, X, 3))
    assert e2 > e3 > 0


def test_bits_bracket_and_interpolation() -> None:
    assert orc.bits_bracket(0.3125) == (2, 2)
    assert orc.bits_bracket(0.4) == (2, 3)
    assert orc.bits_bracket(0.2) == (None, 2)
    u = orc.uniform_at(0.375, {2: 0.4, 3: 0.2})
    assert u["uniform_interpolated"] and u["uniform_err"] == pytest.approx(math.sqrt(0.4 * 0.2))
    assert orc.uniform_at(0.3125, {2: 0.4, 3: 0.2})["uniform_err"] == 0.4
    c = orc.uniform_at(0.1, {2: 0.4, 3: 0.2})
    assert c["uniform_clamped"] and c["uniform_err"] == 0.4


def test_eps_parsing() -> None:
    assert orc.parse_eps("0.10,0.01,0.03,0.03") == (0.01, 0.03, 0.10)
    assert orc.parse_eps("1%,3%") == (0.01, 0.03)
    assert [orc.eps_key(e) for e in (0.01, 0.03, 0.1, 0.005)] == ["0.01", "0.03", "0.10", "0.005"]
    for bad in ("0", "-1", "", "nan"):
        with pytest.raises(ValueError):
            orc.parse_eps(bad)


# --------------------------------------------------------------------------------------- verdict
@pytest.mark.parametrize("args,expect", [
    ((0.20, 0.60, 8.0e9, 1.5, 2.0), "PROMISING"),
    ((0.20, 0.60, 9.0e9, 1.5, 2.0), "WEAK"),           # bytes clause fails, fraction <= 0.5
    ((0.40, 0.10, 12e9, 0.5, 1.0), "WEAK"),
    ((0.90, 0.30, 12e9, 0.5, 1.0), "WEAK"),            # efficiency 0.2-0.5
    ((0.90, 0.10, 12e9, 0.5, 1.1), "NO_ADVANTAGE"),
    ((0.90, 0.90, 12e9, 0.5, 1.5), "NOT_SUPPORTED"),
    ((None, None, None, None, None), "NOT_SUPPORTED"),
])
def test_verdict_rule(args: tuple[Any, ...], expect: str) -> None:
    v, clauses = orc.evaluate_verdict(*args)
    assert v == expect
    assert {"promising_frac_le_0_25", "weak_eff_0_2_to_0_5", "no_advantage_q3_le_1_2x"} <= set(clauses)
    assert orc.VERDICT_RANK["PROMISING"] > orc.VERDICT_RANK["WEAK"] > orc.VERDICT_RANK["NO_ADVANTAGE"] \
        > orc.VERDICT_RANK["NOT_SUPPORTED"]


def test_rule_text_is_the_manifest_text(repo_root: Path) -> None:
    man = json.loads((repo_root / "docs" / "design" / "manifest.json").read_text(encoding="utf-8"))
    e7 = next(e for e in man["experiments"] if str(e.get("id", "")).startswith("E7"))
    assert e7["thresholds"] == orc.RULE_TEXT


# -------------------------------------------------------------------------------------- sanity
def test_sanity_passes_on_synthetic_ffn_out(synth_acts_dir: Any, tiny_dense_gguf: Any) -> None:
    for il in synth_acts_dir.layers:
        Wd = dc.load_matrix(tiny_dense_gguf.path, f"blk.{il}.ffn_down.weight")
        s = orc.sanity_check(Wd, synth_acts_dir.geglu[il], synth_acts_dir.out[il], layer=il)
        assert s["pass"] and s["rel_err_median"] < 0.03 and s["n_tokens"] == 32
        bad = orc.sanity_check(2.0 * Wd, synth_acts_dir.geglu[il], synth_acts_dir.out[il], layer=il)
        assert not bad["pass"] and bad["rel_err_median"] > orc.SANITY_ABORT_MAX


# --------------------------------------------------------------------------------- CLI end to end
@pytest.fixture(scope="module")
def decomp_dir(tmp_path_factory: pytest.TempPathFactory, tiny_dense_gguf: Any) -> Path:
    out = tmp_path_factory.mktemp("decomp")
    rc = dc.main(["--gguf", str(tiny_dense_gguf.path), "--layers", "0,1", "--mats", "gate,up,down",
                  "--bases", "q2,lowrank16", "--group", "32", "--res-bits", "4", "--out", str(out),
                  "--lowrank-backend", "numpy", "--min-free-gb", "0.01", "--log-level", "WARNING"])
    assert rc == 0
    return out


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory, decomp_dir: Path, synth_acts_dir: Any,
        tiny_dense_gguf: Any) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("oracle")
    rc = orc.main(["--decomp", str(decomp_dir), "--acts", str(synth_acts_dir.path), "--gguf",
                   str(tiny_dense_gguf.path), "--tokens", "512", "--eps", "0.01,0.03,0.10",
                   "--out", str(out / "oracle.json"), "--md", str(out / "H2_RESULTS.md"), "--batch", "8",
                   "--no-sha", "--log-level", "WARNING"])
    doc = json.loads((out / "oracle.json").read_text(encoding="utf-8")) if (out / "oracle.json").is_file() else None
    return {"rc": rc, "out": out, "doc": doc}


def test_cli_end_to_end_contract(e2e: dict[str, Any]) -> None:
    assert e2e["rc"] == 0
    doc = e2e["doc"]
    assert {"model", "acts_dir", "decomp_dir", "sampling", "sanity", "per_case", "summary"} <= set(doc)
    samp = doc["sampling"]
    assert {"layers", "mats", "bases", "group_cols", "res_bits", "n_tokens", "eps"} <= set(samp)
    assert samp["eps"] == [0.01, 0.03, 0.1] and samp["layers"] == [0, 1] and samp["n_tokens"] == 32
    assert samp["mats"] == ["gate", "up", "down"] and samp["group_cols"] == 32 and samp["res_bits"] == 4
    assert [s["layer"] for s in doc["sanity"]] == [0, 1]
    for s in doc["sanity"]:
        assert {"layer", "rel_err_median", "rel_err_p95", "pass"} <= set(s) and s["pass"] is True
    cases = doc["per_case"]
    assert len(cases) == 2 * 3 * 2 * 3                      # layers x mats x bases x phases
    assert {c["phase"] for c in cases} == {"all", "prefill", "gen"}
    for c in cases:
        assert {"layer", "mat", "base", "phase", "n_tokens", "bytes_q4_0", "base_bytes", "residual_bytes_total",
                "err_base_mean", "oracle", "curve", "bound", "tightness", "spearman_bound_vs_actual",
                "iso_byte", "seconds"} <= set(c)
        assert set(c["oracle"]) == {"0.01", "0.03", "0.10"} and set(c["bound"]) == {"0.01", "0.03", "0.10"}
        for o in c["oracle"].values():
            assert {"frac_bytes_mean", "frac_bytes_p90", "k_groups_mean"} <= set(o)
        for b in c["bound"].values():
            assert {"certified_frac_bytes_mean", "realised_err_mean", "soundness_violations",
                    "certification_efficiency"} <= set(b)
            assert b["soundness_violations"] == 0
        assert {"p10", "p50", "p90"} <= set(c["tightness"])
        assert all(len(p) == 3 for p in c["curve"])
        for e in c["iso_byte"]:
            assert {"bytes_per_elem", "uniform_err", "base_plus_oracle_err", "base_plus_certified_err"} <= set(e)
        sp = c["spearman_bound_vs_actual"]
        assert sp is None or -1.0 <= sp <= 1.0
    by = {(c["layer"], c["mat"], c["base"], c["phase"]): c["n_tokens"] for c in cases}
    assert by[(0, "down", "q2", "prefill")] + by[(0, "down", "q2", "gen")] == by[(0, "down", "q2", "all")] == 32
    s = doc["summary"]
    assert {"frac_0_03_median", "cert_eff_median", "soundness_violations_total", "implied_bytes_per_token",
            "s_per_token_nvme_850mbps", "tok_s_in_ram_10gbps", "verdict", "rule"} <= set(s)
    assert s["verdict"] in orc.VERDICTS and s["rule"] == orc.RULE_TEXT
    assert s["soundness_violations_total"] == 0 and s["pipeline"] == "VALID"
    imp = s["implied_bytes_per_token"]
    assert {"attention", "head", "ffn_base", "ffn_residual_at_0_03", "total", "kind", "note"} <= set(imp)
    assert imp["kind"] == "projected"
    assert imp["total"] == pytest.approx(imp["attention"] + imp["head"] + imp["ffn_base"] + imp["ffn_residual_at_0_03"])


def test_cli_markdown_and_render_from(e2e: dict[str, Any]) -> None:
    out, doc = e2e["out"], e2e["doc"]
    md = (out / "H2_RESULTS.md").read_text(encoding="utf-8")
    assert f"**Verdict: {doc['summary']['verdict']}**" in md
    assert orc.RULE_TEXT in md and "Pipeline VALID" in md
    for head in ("## Sanity (C6)", "## Oracle residual byte fraction", "## Norm-bound certification",
                 "### Bound tightness", "## Iso-byte comparison (C5", "## Extrapolation to the whole model (PROJECTED)"):
        assert head in md
    assert orc.main(["--render-from", str(out / "oracle.json"), "--md", str(out / "again.md")]) == 0
    assert (out / "again.md").read_text(encoding="utf-8") == md
    # rendering a synthetic document directly also carries the verdict string
    md2 = orc.render_markdown(doc)
    assert f"Verdict: {doc['summary']['verdict']}" in md2


def test_cli_envelopes_are_valid_results(e2e: dict[str, Any]) -> None:
    out = e2e["out"]
    envs = sorted(out.glob("E7_*.json"))
    assert len(envs) == 4                                   # 2 bases x (RAM, NVMe)
    for p in envs:
        env = json.loads(p.read_text(encoding="utf-8"))
        assert rs.validate_result(env) == []
        assert env["experiment"] == "E7" and env["kind"] == "projected" and env["assumptions"]
        assert env["variant"].startswith("h2-") and env["read_bytes_per_token"] > 0
    loaded = rs.load_results(out)
    assert len(loaded) == 4                                 # oracle.json is not an envelope
    assert "## E7" in rs.render_markdown(loaded)


def test_cli_y_ref_recon_exact(tmp_path: Path, synth_acts_dir: Any, tiny_dense_gguf: Any) -> None:
    d = tmp_path / "decomp16"
    assert dc.main(["--gguf", str(tiny_dense_gguf.path), "--layers", "0", "--mats", "gate,down", "--bases", "q2",
                    "--group", "32", "--res-bits", "16", "--out", str(d), "--min-free-gb", "0.01",
                    "--log-level", "WARNING"]) == 0
    out = tmp_path / "o.json"
    rc = orc.main(["--decomp", str(d), "--acts", str(synth_acts_dir.path), "--y-ref", "recon", "--out", str(out),
                   "--no-envelopes", "--log-level", "WARNING"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["sampling"]["y_ref_source"] == "recon"
    for c in doc["per_case"]:
        assert c["floor_err_mean"] < 1e-5
        assert c["oracle"]["0.01"]["unreachable_frac"] == 0.0
    assert doc["summary"]["implied_bytes_per_token"]["mats_extrapolated"] == ["up"]


def test_cli_sanity_abort_and_invalid(tmp_path: Path, decomp_dir: Path, synth_acts_dir: Any,
                                      tiny_dense_gguf: Any) -> None:
    from pigemma import trace as tr
    acts = tr.read_acts_dir(synth_acts_dir.path)
    arrays = {}
    for il in synth_acts_dir.layers:
        for kind in ("ffn_norm", "ffn_geglu", "ffn_out"):
            a = np.asarray(acts.get(il, kind), dtype=np.float32)
            arrays[(il, kind)] = 3.0 * a if kind == "ffn_out" else a
    bad = tmp_path / "acts_bad"
    tr.write_acts_dir(bad, arrays, phase=acts.phase, token_ids=acts.token_ids,
                      meta={"gguf_path": str(tiny_dense_gguf.path)})
    out = tmp_path / "out" / "oracle.json"
    args = ["--decomp", str(decomp_dir), "--acts", str(bad), "--gguf", str(tiny_dense_gguf.path), "--out", str(out),
            "--no-sha", "--log-level", "WARNING"]
    assert orc.main(args) == 1 and not out.exists()        # abort before any heavy work
    assert orc.main(args + ["--no-sanity-abort", "--no-envelopes"]) == 1   # pipeline INVALID, still written
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["summary"]["pipeline"] == "INVALID" and doc["summary"]["verdict_void"] is True


def test_cli_wrong_gguf_is_detected(tmp_path: Path, decomp_dir: Path, synth_acts_dir: Any) -> None:
    import synth
    other = synth.write_tiny_dense_gguf(tmp_path / "other.gguf", seed=7)
    rc = orc.main(["--decomp", str(decomp_dir), "--acts", str(synth_acts_dir.path), "--gguf", str(other.path),
                   "--out", str(tmp_path / "o.json"), "--no-sha", "--no-sanity-abort", "--log-level", "WARNING"])
    assert rc == 1 and not (tmp_path / "o.json").exists()


def test_cli_exit_codes(tmp_path: Path, decomp_dir: Path, synth_acts_dir: Any) -> None:
    acts = str(synth_acts_dir.path)
    o = str(tmp_path / "o.json")
    assert orc.main([]) == 2
    assert orc.main(["--decomp", str(decomp_dir), "--acts", acts, "--mats", "gate,bogus", "--out", o]) == 2
    assert orc.main(["--decomp", str(decomp_dir), "--acts", acts, "--eps", "0", "--out", o]) == 2
    assert orc.main(["--decomp", str(decomp_dir), "--acts", acts, "--tokens", "0", "--out", o]) == 2
    assert orc.main(["--decomp", str(tmp_path / "nope"), "--acts", acts, "--out", o]) == 3
    assert orc.main(["--decomp", str(decomp_dir), "--acts", str(tmp_path / "nope"), "--out", o]) == 3
    assert orc.main(["--decomp", str(decomp_dir), "--acts", acts, "--gguf", str(tmp_path / "missing.gguf"),
                     "--out", o]) == 3
    assert orc.main(["--decomp", str(decomp_dir), "--acts", acts, "--layers", "9", "--out", o]) == 3
    assert orc.main(["--render-from", str(tmp_path / "missing.json"), "--md", str(tmp_path / "x.md")]) == 3
    assert not Path(o).exists()


def test_discover_cases_filters(decomp_dir: Path) -> None:
    allc = orc.discover_cases(decomp_dir)
    assert len(allc) == 12 and allc[0].layer == 0 and allc[0].mat == "gate"
    sel = orc.discover_cases(decomp_dir, layers=[1], mats=["down"], bases=["q2"])
    assert [(c.layer, c.mat, c.base) for c in sel] == [(1, "down", "q2")]


# ------------------------------------------------------------------------------- integration (31B)
@pytest.mark.integration
def test_real_31b_down_matrix_pipeline(tmp_path: Path, require_model: Any, repo_root: Path) -> None:
    """Real Q4_0 weights (one tensor read through map31.json, no inference), synthetic activations.

    Checks the real-file path end to end: decompose L20.down q2, an acts dir whose ffn_out is
    W_down_q4 @ geglu, then the oracle CLI with --map: sanity passes, the npz/GGUF consistency
    check passes, 0 soundness violations, and the q2 base / uniform q3 errors match the values
    measured on this matrix in docs/HYPOTHESES.md 5.5 / 5.7 (0.383 and 0.210).
    """
    from pigemma import trace as tr
    require_model("31b-q4")
    map31 = repo_root / "traces" / "h2" / "map31.json"
    if not map31.is_file():
        pytest.skip("traces/h2/map31.json not built")
    d = tmp_path / "decomp"
    assert dc.main(["--map", str(map31), "--layers", "20", "--mats", "down", "--bases", "q2", "--group", "128",
                    "--res-bits", "4", "--out", str(d), "--log-level", "WARNING"]) == 0
    W = dc.load_matrix(map31, "blk.20.ffn_down.weight")
    rng = np.random.default_rng(0)
    T = 64
    a = rng.standard_normal((T, W.shape[1])).astype(np.float32)
    g = (0.5 * a * (1 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3))) * rng.standard_normal(a.shape))
    g = g.astype(np.float16).astype(np.float32)
    y = (g.astype(np.float64) @ W.T.astype(np.float64)).astype(np.float32)
    acts = tmp_path / "acts"
    tr.write_acts_dir(acts, {(20, "ffn_geglu"): g, (20, "ffn_out"): y},
                      phase=(np.arange(T) >= 48).astype(np.uint8), meta={"model": "gemma-4-31B_q4_0-it.gguf"})
    del W, g, y, a
    out = tmp_path / "oracle.json"
    assert orc.main(["--decomp", str(d), "--acts", str(acts), "--map", str(map31), "--out", str(out),
                     "--no-envelopes", "--no-sha", "--log-level", "WARNING"]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["summary"]["pipeline"] == "VALID" and doc["summary"]["soundness_violations_total"] == 0
    assert doc["sanity"][0]["rel_err_median"] < 1e-3
    c = next(x for x in doc["per_case"] if x["phase"] == "all")
    assert c["err_base_mean"] == pytest.approx(0.383, abs=0.02)
    q3 = next(e for e in c["iso_byte"] if e["label"] == "q3")
    assert q3["uniform_err"] == pytest.approx(0.210, abs=0.02)
    assert doc["summary"]["implied_bytes_per_token"]["attention"] == orc.ATTENTION_BYTES_31B
