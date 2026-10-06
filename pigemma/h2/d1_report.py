"""pigemma.h2.d1_report - apply the pre-registered D1 rule to results/dense/L*.json and emit tables.

    python -m pigemma.h2.d1_report --dir results/dense [--runtime results/dense/runtime.json]

Writes ``<dir>/d1_summary.json`` (verdict, per-layer clause values, projections) and
``<dir>/d1_tables.md`` (markdown tables used by docs/results/DENSE_FEASIBILITY.md). Every number is
read from the per-layer JSON (MEASURED) except the keys under ``projection`` (PROJECTED from the
sampled layers to all 60 layers, with the stated ASSUMED constants).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

N_LAYERS = 60
ATTN_BYTES = 4_770_160_640       # EXACT, 31B Q4_0 GGUF (traces/h2/map31.json, E7)
HEAD_BYTES = 1_156_076_544       # EXACT
NVME_MBPS = 850.0                # ASSUMED: Pi 5 NVMe Gen3 x1 (published 784-894 MB/s)
SD_MBPS = 91.6                   # MEASURED on the real Pi (microSD, O_DIRECT 4 MiB)
PI_RAM_GBPS = 10.0               # ASSUMED: ~10 GB/s effective in llama.cpp decode


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def table(head: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out + [""]


def best_at_target(comp: dict[str, Any], org: str, target: str = "0.01", need: float = 0.9) -> dict[str, Any] | None:
    """Cheapest per-matrix eps grid point whose FFN error <= target on >= need of the tokens."""
    best = None
    for eps, ent in comp["per_matrix_eps"].items():
        e = ent[org]
        if e["frac_tokens_ffn_err_le"][target] >= need:
            if best is None or e["total_frac_of_q4_0"] < best["total_frac_of_q4_0"]:
                best = {"eps_per_matrix": float(eps), **e}
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/dense")
    ap.add_argument("--runtime", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    layers = sorted((json.loads(p.read_text(encoding="utf-8")) for p in d.glob("L*.json")), key=lambda x: x["layer"])
    pre = json.loads((d / "preregistration.json").read_text(encoding="utf-8"))
    md: list[str] = []
    summ: dict[str, Any] = {"generator": "pigemma.h2.d1_report", "preregistration": pre["decision_rule"],
                            "layers": [L["layer"] for L in layers], "per_layer": {}}

    # ---- runtime floor + uniform baselines ----
    rows = []
    for L in layers:
        rf = L["runtime_floor"]
        rows.append([L["layer"], L["tokens"]["calibration"], f"{L['tokens']['heldout']} ({L['tokens']['heldout_gen']} gen)",
                     f(rf["ffn_out_captured_vs_f32_composed_from_captured_ffn_norm"]["median"]),
                     f(rf["ffn_out_captured_vs_f32_composed_from_captured_ffn_norm"]["p90"]),
                     f(rf["geglu_captured_vs_f32_from_captured_ffn_norm"]["median"]),
                     f(rf["ffn_out_captured_vs_f32_down_of_captured_geglu"]["median"])])
    md += ["### Runtime numerical floor (MEASURED)", "",
           "llama.cpp's own FFN output (captured, CPU repack kernels, Q8_0-quantised activations) vs float32 math on the same dequantised Q4_0 weights; relative L2 per held-out token.", ""]
    md += table(["layer", "cal tokens", "held-out tokens", "FFN out median", "FFN out p90", "geglu median", "down-only median"], rows)

    rows = []
    for L in layers:
        for qn, lab in (("q3_k_imatrix", "Q3_K (imatrix) 3.44 bpw"), ("q3_absmax", "q3 absmax 3.5 bpw"), ("q2_absmax", "q2 absmax 2.5 bpw")):
            u = L["uniform"][qn]
            rows.append([L["layer"], lab, f"{u['ffn_bytes'] / L['bytes']['q4_0_ffn']:.3f}",
                         f(u["matrices"]["gate"]["rel_err"]["mean"]), f(u["matrices"]["up"]["rel_err"]["mean"]),
                         f(u["matrices"]["down"]["rel_err"]["mean"]), f(u["ffn_rel_err"]["mean"]), f(u["ffn_rel_err"]["p90"])])
    md += ["### Uniform re-quantisation of the same layer (MEASURED, held-out)", ""]
    md += table(["layer", "format", "bytes / Q4_0", "gate err", "up err", "down err", "FFN err mean", "FFN err p90"], rows)

    # ---- isolated: base cost/error ----
    rows = []
    for L in layers:
        for key, e in L["isolated"].items():
            kind, r, m = key.split("/")
            rows.append([L["layer"], kind, r[1:], m, f"{e['base_frac_of_q4_0']:.3f}", f(e["base_rel_err"]["mean"], 3),
                         f"{e['base_rel_err_q8_factors_first_batch']['mean']:.3f} / {e['base_rel_err_fp32_factors_first_batch']['mean']:.3f}",
                         f(L["uniform"]["q3_k_imatrix"]["matrices"][m]["rel_err"]["mean"], 3),
                         f(e["thresholds"]["0.01"]["col_oracle_frac"]["mean"], 3),
                         f(e["thresholds"]["0.01"]["row_oracle_frac"]["mean"], 3),
                         f(e["thresholds"]["0.01"]["col_online_frac"]["mean"], 3),
                         f(e["thresholds"]["0.01"].get("col_static_fixed_frac_for_90pct_tokens"), 3),
                         f(e["thresholds"]["0.01"].get("row_static_fixed_frac_for_90pct_tokens"), 3),
                         f(e["col_oracle_tokens_where_partial_reading_exceeds_base_err_frac"], 2),
                         f"{e['reconstruction_rel_err_all_blocks_first_batch']['max']:.1e}"])
    md += ["### Isolated matrices: base cost, base error, residual blocks needed for 1e-2 (MEASURED, held-out, perfect inputs)", "",
           "Fractions are of the matrix's residual blocks (= of its Q4_0 bytes). 'hump' = share of tokens whose error rises above the base error at some point while column blocks are added in the greedy-oracle order (corrections that cancel).", ""]
    md += table(["layer", "base", "rank", "matrix", "base bytes / Q4_0", "base err", "base err 1st batch q8 / fp32 factors", "Q3_K err",
                 "col-oracle frac @1e-2", "row-oracle frac @1e-2", "col-online frac @1e-2", "col-static frac (90% tok) @1e-2", "row-static frac (90% tok) @1e-2", "hump", "recon err (all blocks)"], rows)

    # ---- isolated equal bytes ----
    rows = []
    for L in layers:
        for key, e in L["isolated"].items():
            kind, r, m = key.split("/")
            q = e["equal_bytes"]["q3_k_imatrix"]
            rows.append([L["layer"], kind, r[1:], m, f(e["base_rel_err"]["mean"]), f(q["uniform_err"]["mean"]),
                         f(q["col_oracle_err"]["mean"]) if q["col_oracle_err"] else "base > budget",
                         f(q["row_oracle_err"]["mean"]) if q["row_oracle_err"] else "base > budget",
                         f(q["col_online_err"]["mean"]) if q["col_online_err"] else "base > budget",
                         f(q["col_static_err"]["mean"]) if q.get("col_static_err") else "base > budget",
                         f(q["row_static_err"]["mean"]) if q.get("row_static_err") else "base > budget"])
    md += ["### Isolated matrices at Q3_K's bytes (MEASURED, held-out)", "",
           "Total bytes (base + residual + metadata) = Q3_K's bytes for that matrix; cells are mean relative output errors. col-oracle = best prefix of the greedy oracle order; row-oracle = optimal per token; col-online = norm-bound order (decides from resident norms and ||x_g|| only), exactly k blocks; col-/row-static = one fixed block set ranked on calibration energy.", ""]
    md += table(["layer", "base", "rank", "matrix", "base alone err", "Q3_K err", "col-oracle", "row-oracle", "col-online", "col-static", "row-static"], rows)

    # ---- composed ----
    rows, rows2 = [], []
    verdict_layers = {}
    for L in layers:
        lay: dict[str, Any] = {"iso_byte": [], "few": [], "q3k_ffn_err": L["uniform"]["q3_k_imatrix"]["ffn_rel_err"]["mean"]}
        for rk, c in L["composed"].items():
            q = c["equal_bytes"]["q3_k_imatrix"]
            for org, k2 in (("col", "col_err"), ("row", "row_err"), ("col-online", "col_online_err"),
                            ("col-static", "col_static_err"), ("row-static", "row_static_err")):
                val = q.get(k2, {}).get("mean") if q["base_fits"] else None
                rows.append([L["layer"], rk[1:], org, f"{c['base_frac_of_q4_0_ffn']:.3f}", f"{q['residual_frac']:.3f}",
                             f(q["uniform_ffn_err"]["mean"]), f(val),
                             f(q["uniform_ffn_err"]["mean"] / val, 2) if val else "n/a"])
                if org in ("col", "row") and val:
                    lay["iso_byte"].append({"rank": c["rank"], "org": org, "err": val, "q3k": q["uniform_ffn_err"]["mean"],
                                            "ratio_q3k_over_oracle": q["uniform_ffn_err"]["mean"] / val})
            for org in ("col", "row"):
                for tgt in ("0.01", "0.005", "0.001"):
                    b = best_at_target(c, org, tgt)
                    rows2.append([L["layer"], rk[1:], org, tgt, f"{c['base_frac_of_q4_0_ffn']:.3f}",
                                  f"{b['eps_per_matrix']:g}" if b else "none",
                                  f(b["residual_frac_of_q4_0"], 3) if b else "> grid",
                                  f(b["total_frac_of_q4_0"], 3) if b else "> grid",
                                  f(b["ffn_rel_err"]["mean"]) if b else "n/a"])
                    if tgt == "0.01":
                        lay["few"].append({"rank": c["rank"], "org": org, "base_frac": c["base_frac_of_q4_0_ffn"],
                                           "residual_frac": b["residual_frac_of_q4_0"] if b else None,
                                           "total_frac": b["total_frac_of_q4_0"] if b else None,
                                           "eps_per_matrix": b["eps_per_matrix"] if b else None})
        best_iso = max(lay["iso_byte"], key=lambda z: z["ratio_q3k_over_oracle"]) if lay["iso_byte"] else None
        lay["best_iso_byte"] = best_iso
        lay["ISO_BYTE_WIN"] = bool(best_iso and best_iso["err"] <= 0.5 * best_iso["q3k"])
        lay["INCREMENTAL_RANGE"] = bool(best_iso and best_iso["ratio_q3k_over_oracle"] >= 1.2)
        fe = [z for z in lay["few"] if z["residual_frac"] is not None and z["base_frac"] <= 0.25]
        lay["best_few"] = min(fe, key=lambda z: z["total_frac"]) if fe else None
        lay["min_residual_frac_at_1e-2"] = min((z["residual_frac"] for z in lay["few"] if z["residual_frac"] is not None), default=None)
        lay["FEW_CORRECTIONS"] = bool(any(z["residual_frac"] <= 0.15 for z in fe))
        verdict_layers[L["layer"]] = lay
    md += ["### Composed FFN at Q3_K's bytes (MEASURED, held-out, captured layer input)", "",
           "FFN = W_down(gelu_tanh(W_gate x) * W_up x) with all three matrices approximated; the same residual fraction on each matrix so that base + residual + metadata = Q3_K's FFN bytes. ratio > 1 means the representation beats Q3_K.", ""]
    md += table(["layer", "rank", "blocks", "base / Q4_0", "residual frac", "Q3_K FFN err", "base+oracle FFN err", "Q3_K err / ours"], rows)
    md += ["### Composed FFN: bytes needed for an FFN-output tolerance on >= 90% of held-out tokens (MEASURED; oracle selection)", "",
           "Searched over a per-matrix tolerance grid (0.2 ... 0.001, same on all three matrices); 'total / Q4_0' includes the base.", ""]
    md += table(["layer", "rank", "blocks", "FFN target", "base / Q4_0", "per-matrix eps", "residual / Q4_0", "total / Q4_0", "FFN err mean"], rows2)
    rows3 = []
    for L in layers:
        for rk, c in L["composed"].items():
            for fk, e in c.get("static_fixed_fraction", {}).items():
                rows3.append([L["layer"], rk[1:], fk, f(e["total_frac_of_q4_0"], 3),
                              f(e["col_static"]["ffn_rel_err"]["mean"]), f(e["col_static"]["frac_tokens_ffn_err_le"]["0.01"], 2),
                              f(e["row_static"]["ffn_rel_err"]["mean"]), f(e["row_static"]["frac_tokens_ffn_err_le"]["0.01"], 2)])
    md += ["### Composed FFN with STATIC correction selection (MEASURED, held-out)", "",
           "One fixed set of blocks per matrix (ranked by mean calibration energy), the same for every token; residual fraction 0 = the base alone. Compare with the uniform rows above at similar total bytes (Q3_K = 0.764, q2 = 0.556 of Q4_0).", ""]
    md += table(["layer", "rank", "residual frac", "total / Q4_0", "col-static FFN err", "col-static tokens <= 1e-2", "row-static FFN err", "row-static tokens <= 1e-2"], rows3)

    # ---- verdict ----
    allL = list(verdict_layers.values())
    iso = all(z["ISO_BYTE_WIN"] for z in allL)
    few = all(z["FEW_CORRECTIONS"] for z in allL)
    inc = all(z["INCREMENTAL_RANGE"] for z in allL)
    if iso and few:
        verdict = "PROMISING"
    elif inc and not few:
        verdict = "INCREMENTAL"
    else:
        verdict = "NEGATIVE"
    # failure attribution
    reasons = []
    for L in layers:
        lay = verdict_layers[L["layer"]]
        q3 = {m: L["uniform"]["q3_k_imatrix"]["matrices"][m]["rel_err"]["mean"] for m in ("gate", "up", "down")}
        cheap = [e for k, e in L["isolated"].items() if k.startswith("whiten/") and e["base_frac_of_q4_0"] <= 0.25]
        worse = [e for e in cheap if e["base_rel_err"]["mean"] > q3[e["matrix"]]]
        lay["affordable_bases_worse_than_q3k"] = f"{len(worse)}/{len(cheap)}"
    if not few:
        mins = [z["min_residual_frac_at_1e-2"] for z in allL]
        if all(m is None or m > 0.5 for m in mins):
            reasons.append("residuals too dense: at FFN error 1e-2 the oracle still needs > 50% of the Q4_0 residual bytes on every sampled layer (min over ranks/organisations: "
                           + ", ".join("none on grid" if m is None else f"{m:.2f}" for m in mins) + ")")
    if not iso:
        reasons.append("no iso-byte win: at Q3_K's bytes the oracle-selected base+residual is not 2x better than Q3_K (best ratio per layer: "
                       + ", ".join(f"{z['best_iso_byte']['ratio_q3k_over_oracle']:.2f}" if z["best_iso_byte"] else "n/a" for z in allL) + ")")
    reasons.append("base too expensive / too weak: affordable whitened bases (<= 0.25x Q4_0) with error above Q3_K's: "
                   + ", ".join(f"L{L['layer']} {verdict_layers[L['layer']]['affordable_bases_worse_than_q3k']}" for L in layers))
    summ["verdict"] = verdict
    summ["clauses"] = {"ISO_BYTE_WIN_all_layers": iso, "FEW_CORRECTIONS_all_layers": few, "INCREMENTAL_RANGE_all_layers": inc}
    summ["failure_attribution"] = reasons
    summ["per_layer"] = verdict_layers

    # ---- projection (PROJECTED) ----
    proj = {}
    for L in layers:
        pass
    q4_ffn = sum(L["bytes"]["q4_0_ffn"] for L in layers) / len(layers)
    q3k_ffn = sum(L["uniform"]["q3_k_imatrix"]["ffn_bytes"] for L in layers) / len(layers)
    proj["q4_0_ffn_GB_per_token_60_layers"] = N_LAYERS * q4_ffn / 1e9
    proj["q3_k_ffn_GB_per_token_60_layers"] = N_LAYERS * q3k_ffn / 1e9
    proj["attention_plus_head_GB_per_token"] = (ATTN_BYTES + HEAD_BYTES) / 1e9
    cand = []
    for L in layers:
        lay = verdict_layers[L["layer"]]
        for z in lay["few"]:
            if z["total_frac"] is not None:
                cand.append((L["layer"], z))
    # cheapest config reaching 1e-2 per layer, averaged
    per_layer_best = {}
    for L in layers:
        zs = [z for z in verdict_layers[L["layer"]]["few"] if z["total_frac"] is not None]
        if zs:
            per_layer_best[L["layer"]] = min(zs, key=lambda z: z["total_frac"])
    if len(per_layer_best) == len(layers):
        base_frac = sum(z["base_frac"] for z in per_layer_best.values()) / len(layers)
        res_frac = sum(z["residual_frac"] for z in per_layer_best.values()) / len(layers)
        proj["at_ffn_1e-2"] = {
            "base_resident_GB_60_layers": N_LAYERS * q4_ffn * base_frac / 1e9,
            "residual_streamed_GB_per_token_60_layers": N_LAYERS * q4_ffn * res_frac / 1e9,
            "total_moved_GB_per_token_incl_attention_head": N_LAYERS * q4_ffn * (base_frac + res_frac) / 1e9 + (ATTN_BYTES + HEAD_BYTES) / 1e9,
            "s_per_token_streaming_residual_nvme_850MBps_assumed": N_LAYERS * q4_ffn * res_frac / (NVME_MBPS * 1e6),
            "s_per_token_streaming_residual_microSD_91.6MBps_measured_bw": N_LAYERS * q4_ffn * res_frac / (SD_MBPS * 1e6),
            "configs": {str(k): v for k, v in per_layer_best.items()},
        }
    proj["q4_0_all_in_ram_tok_s_at_10GBps_assumed"] = PI_RAM_GBPS * 1e9 / (N_LAYERS * q4_ffn + ATTN_BYTES + HEAD_BYTES)
    proj["note"] = "PROJECTED: sampled layers 2/30/57 assumed representative of all 60; bandwidth-only; ignores compute and the base/selection overheads"
    summ["projection"] = proj
    if a.runtime and Path(a.runtime).is_file():
        summ["runtime"] = json.loads(Path(a.runtime).read_text(encoding="utf-8"))
    (d / "d1_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    md = [f"Verdict (pre-registered rule): **{verdict}**", ""] + [f"- {r}" for r in reasons] + [""] + md
    (d / "d1_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(verdict)
    for r in reasons:
        print(" -", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
