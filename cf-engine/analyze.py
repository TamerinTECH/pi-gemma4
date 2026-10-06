#!/usr/bin/env python3
"""Analyze CF_REPORT/probe evidence offline; never runs a model.

Example:
  python cf-engine/analyze.py --reference-report base.report.json \
    --candidate-report cf.report.json --reference-requests base.requests.jsonl \
    --candidate-requests cf.requests.jsonl --reference-logits base.logits.bin \
    --candidate-logits cf.logits.bin --out comparison.json

Logits are repeated <QQQ headers (vocabulary, request, token), followed by
vocabulary little-endian float32 values. NumPy is needed only for logits.
Exit 0: requested comparisons passed (or profile only); 1: comparison failed;
2: invalid/truncated input. Thresholds below are fixed, not command-line knobs.
"""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


# Preregistered 2026-10-06 before candidate end-to-end logits were inspected.
THRESHOLDS = {"relative_l2": 2e-4, "max_absolute": 2e-3,
              "kl_reference_to_candidate": 1e-5, "all_greedy_ids_equal": True}


def number(value, field, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{field}: invalid/nonfinite number")
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def category(row):
    name = row["name"].lower()
    op = row["op"]
    weight = row.get("weight_name", "").lower()
    if op == "MUL_MAT_ID":
        if "down" in weight or "down" in name:
            return "moe_down"
        if "gate" in weight or "up" in weight or "gate" in name or "up" in name:
            return "moe_gate_up"
        return "moe_projection_other"
    if op == "MUL_MAT" and weight in ("token_embd.weight", "output.weight"):
        return "output_head"
    if op == "MUL_MAT" and ".ffn_" in weight:
        return "moe_router" if "gate_inp" in weight else "dense_ffn"
    if op == "MUL_MAT" and ".attn_" in weight:
        return "attention"
    if "result_output" in name or name in ("logits", "output", "output_softcap"):
        return "output_head"
    if "ffn_moe" in name:
        if any(s in name for s in ("logits", "probs", "topk", "argsort", "weights")):
            return "moe_router"
        return "moe_nonlinear_combine_other"
    if "ffn" in name:
        return "dense_ffn"
    if any(s in name for s in ("attn", "kq", "qcur", "kcur", "vcur", "rope", "rot_k", "rot_v")):
        return "attention"
    if "per_layer" in name or "embd" in name or op == "GET_ROWS":
        return "embedding_lookup_ple"
    return "other"


def summarize_report(path):
    raw = load_json(path)
    if raw.get("format") != "cf-engine-run" or raw.get("version") != 1:
        raise ValueError(f"{path}: unsupported CF report format/version")
    groups = {}
    nodes = []
    dispatch_totals = defaultdict(lambda: {"calls": 0, "elapsed_ms": 0.0})
    for row in raw["ops"]:
        kind = category(row)
        key = (row["phase"], kind, row["dispatch"], row["reason"])
        group = groups.setdefault(key, dict(zip(("phase", "category", "dispatch", "reason"), key)))
        for field in ("calls", "elapsed_ms", "prepare_ms", "multiply_ms", "logical_weight_bytes"):
            value = number(row[field], f"{path}:{row['name']}:{field}")
            group[field] = group.get(field, 0) + value
        dispatch_totals[row["dispatch"]]["calls"] += row["calls"]
        dispatch_totals[row["dispatch"]]["elapsed_ms"] += row["elapsed_ms"]
        nodes.append({**row, "category": kind})
    graph_ms = sum(g["elapsed_ms"] for g in groups.values())
    init_ms = number(raw.get("init_and_identity_check_ms", 0), "init_and_identity_check_ms")
    source_ms = number(raw.get("first_use_source_tensor_identity_ms", 0), "first_use_source_tensor_identity_ms")
    if source_ms > graph_ms + 0.001:
        raise ValueError(f"{path}: source-check time exceeds all graph time")
    validated = raw.get("validation", {}).get("enabled", False)
    cf_calls = sum(v["calls"] for k, v in dispatch_totals.items() if k.startswith("cf"))
    ref_ms = dispatch_totals.get("reference", {}).get("elapsed_ms", 0)
    phases = {}
    for phase in sorted({g["phase"] for g in groups.values()}):
        selected = [g for g in groups.values() if g["phase"] == phase]
        elapsed = sum(g["elapsed_ms"] for g in selected)
        ref = sum(g["elapsed_ms"] for g in selected if g["dispatch"] == "reference")
        phases[phase] = {"graph_ms": elapsed, "reference_runtime_share":
                         None if validated or not elapsed else ref / elapsed}
    return {
        "path": str(path), "mode": raw.get("mode"),
        "execution_contract": raw.get("execution_contract"),
        "kernel_requested": raw.get("kernel_requested"), "kernel_executed": raw.get("kernel_executed"),
        "cf_calls": cf_calls, "cf_executed": cf_calls > 0,
        "dispatch_totals": dict(dispatch_totals), "phases": phases,
        "reference_runtime_share": None if validated or not graph_ms else ref_ms / graph_ms,
        "setup": {"init_and_identity_check_ms": init_ms,
                  "first_use_source_tensor_identity_ms": source_ms,
                  "initialization_in_graph_timing": False,
                  "first_use_source_check_in_graph_timing": True,
                  "source_check_per_phase_attribution": "unavailable in report v1",
                  "graph_ms_minus_first_source_check": graph_ms - source_ms,
                  "adjustment_is_measured_warm_run": False},
        "graph_ms": graph_ms,
        "groups": sorted(groups.values(), key=lambda g: (g["phase"], -g["elapsed_ms"], g["category"])),
        "top_nodes": sorted(nodes, key=lambda r: -r["elapsed_ms"])[:30],
        "memory": {k: raw.get(k) for k in ("memory_policy", "mapped_packed_bytes", "source_file_bytes",
                   "scratch_allocated_bytes", "scratch_active_peak_bytes", "peak_rss_kib", "major_faults")},
        "validation": raw.get("validation"), "proc": raw.get("proc"),
        "timing_note": raw.get("timing_note"), "traffic_note": raw.get("traffic_note"),
        "limitations": ["Fused operations are attributed to their leading node; categories are node-name based.",
                        "Phase label comes from reference GET_ROWS batch size; one-token prefill may be labeled decode.",
                        "Logical weight bytes are modeled accesses, never measured DRAM transactions.",
                        "Validation mode computes reference outputs and cannot measure standalone CF inference speed."]}


def visible_text(row):
    """Never infer visible output from stripped probe text or an approximate token count."""
    if isinstance(row.get("content"), str) and "reasoning_content" in row:
        return {"status": "structured_channels", "text": row["content"],
                "reasoning_chars": len(row.get("reasoning_content") or "")}
    text = row.get("text", "")
    # Complete, anchored Gemma thought channel only. Unknown/nested tags fail closed.
    match = re.fullmatch(r"(?:<start_of_turn>model\n)?<\|channel>thought\r?\n(.*?)<channel\|>(.*?)(?:<end_of_turn>)?",
                         text, flags=re.DOTALL)
    if match and not re.search(r"<\||<channel\||<start_of_turn>|<end_of_turn>", "".join(match.groups())):
        return {"status": "parsed_gemma_thought_channel", "text": match.group(2),
                "reasoning_chars": len(match.group(1))}
    return {"status": "unknown_channels_probe_may_strip_special_tokens", "text": None,
            "reasoning_chars": None}


def compare_profiles(reference, candidate):
    def collect(report):
        totals = defaultdict(lambda: {"calls": 0, "elapsed_ms": 0.0})
        for group in report["groups"]:
            key = (group["phase"], group["category"])
            totals[key]["calls"] += group["calls"]
            totals[key]["elapsed_ms"] += group["elapsed_ms"]
        return totals
    left, right = collect(reference), collect(candidate)
    rows = []
    for phase, kind in sorted(left.keys() | right.keys()):
        a, b = left[(phase, kind)], right[(phase, kind)]
        rows.append({"phase": phase, "category": kind,
                     "reference_ms": a["elapsed_ms"], "candidate_ms": b["elapsed_ms"],
                     "reference_calls": a["calls"], "candidate_calls": b["calls"],
                     "same_leading_op_call_count": a["calls"] == b["calls"],
                     "time_ratio_candidate_over_reference": b["elapsed_ms"]/a["elapsed_ms"] if a["elapsed_ms"] else None})
    return {"groups": rows,
            "note": "Descriptive only: includes first-use source checks and instrumentation; compare workload, calls, execution contracts and repeated end-to-end requests before a speed claim."}


def read_requests(path):
    rows = []
    seen = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["chunk_id"], row["id"])
        if key in seen or "error" in row:
            raise ValueError(f"{path}:{line_no}: duplicate request key/error record")
        seen.add(key)
        ids = row["gen_tokens"]
        if not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ids):
            raise ValueError(f"{path}:{line_no}: invalid generated token IDs")
        if row["n_sampled"] != len(ids) or len(row["sampled_ms"]) != len(ids):
            raise ValueError(f"{path}:{line_no}: inconsistent token/timestamp counts")
        for field in ("t_prefill_ms", "t_total_ms"):
            number(row[field], field)
        times = [number(t, "sampled_ms") for t in row["sampled_ms"]]
        if times != sorted(times) or row["t_prefill_ms"] > row["t_total_ms"] + 0.002:
            raise ValueError(f"{path}:{line_no}: invalid timing order")
        if times and times[-1] > row["t_total_ms"] + 0.002:
            raise ValueError(f"{path}:{line_no}: sampling timestamp after request end")
        if not isinstance(row["text"], str):
            raise ValueError(f"{path}:{line_no}: text is not a string")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no requests")
    return rows


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize_requests(path, rows):
    out = []
    for row in rows:
        times = row["sampled_ms"]
        gaps = [b-a for a, b in zip(times, times[1:])]
        decode_ms = times[-1]-times[0] if len(times) > 1 else 0
        visible = visible_text(row)
        out.append({"key": [row["chunk_id"], row["id"]], "stop": row["stop"],
                    "sampled_model_tokens": len(row["gen_tokens"]),
                    "evaluated_generation_tokens": row["n_gen_evaluated"],
                    "prefill_tokens": row["n_prefill"], "reused_prefix_tokens": row["n_keep"],
                    "prefill_ms": row["t_prefill_ms"], "total_ms": row["t_total_ms"],
                    "first_sample_ms": times[0] if times else None,
                    "prefill_tok_s": row["n_prefill"]*1000/row["t_prefill_ms"] if row["t_prefill_ms"] else None,
                    "decode_interval_tok_s": len(gaps)*1000/decode_ms if decode_ms else None,
                    "full_request_model_tok_s": len(times)*1000/row["t_total_ms"] if row["t_total_ms"] else None,
                    "inter_sample_ms": {"median": statistics.median(gaps) if gaps else None,
                                        "p95": percentile(gaps, .95), "maximum": max(gaps) if gaps else None},
                    "visible": visible, "visible_answer_tok_s": None, "first_visible_answer_ms": None,
                    "text": row["text"]})
    total_ms = sum(r["t_total_ms"] for r in rows)
    total_tokens = sum(r["n_sampled"] for r in rows)
    return {"path": str(path), "request_count": len(rows), "requests": out,
            "total_sampled_model_tokens": total_tokens, "sum_request_ms": total_ms,
            "sequential_request_model_tok_s": total_tokens*1000/total_ms if total_ms else None,
            "aggregate_parallel_tok_s": None,
            "notes": ["Decode uses (sampled tokens - 1)/(last sample - first sample), not n/(n-1 intervals).",
                      "Sampled model tokens may include stop/control/hidden-reasoning tokens.",
                      "Probe times exclude model load and adapter initialization. First-use source checks remain included.",
                      "No per-piece text/timestamp mapping: visible token rate and first visible answer time are unavailable.",
                      "Sum of request durations is valid only for this sequential probe, not concurrent aggregate throughput."]}


def compare_requests(reference, candidate):
    left = {(r["chunk_id"], r["id"]): r for r in reference}
    right = {(r["chunk_id"], r["id"]): r for r in candidate}
    if left.keys() != right.keys():
        return {"passed": False, "error": "different request keys",
                "reference_keys": [list(k) for k in left], "candidate_keys": [list(k) for k in right]}
    comparisons = []
    for key, a in left.items():
        b = right[key]
        config_fields = ("n_prompt_tokens", "n_keep", "n_prefill", "max_tokens", "server_stop", "prefill_calls")
        differences = [k for k in config_fields if a.get(k) != b.get(k)]
        tokens_equal = a["gen_tokens"] == b["gen_tokens"]
        text_equal = a["text"] == b["text"]
        mismatch = next((i for i, (x, y) in enumerate(itertools.zip_longest(a["gen_tokens"], b["gen_tokens"])) if x != y), None)
        comparisons.append({"key": list(key), "workload_field_differences": differences,
                            "token_ids_exact": tokens_equal, "text_exact": text_equal,
                            "first_mismatching_token_index": mismatch, "stop_equal": a["stop"] == b["stop"],
                            "tokens_compared": len(a["gen_tokens"]),
                            "reference_total_ms": a["t_total_ms"], "candidate_total_ms": b["t_total_ms"],
                            "total_latency_ratio_candidate_over_reference": b["t_total_ms"]/a["t_total_ms"] if a["t_total_ms"] else None,
                            "passed": not differences and tokens_equal and text_equal and a["stop"] == b["stop"] and bool(a["gen_tokens"])})
    return {"passed": all(r["passed"] for r in comparisons), "requests": comparisons,
            "limitation": "Prompt token IDs/model hashes are not in requests JSONL; matched counts are not proof of identical input/model."}


def logits_records(path, np):
    seen = set()
    with open(path, "rb") as stream:
        while True:
            header = stream.read(24)
            if not header:
                return
            if len(header) != 24:
                raise ValueError(f"{path}: truncated logits header")
            vocab, request, token = struct.unpack("<QQQ", header)
            if not 1 <= vocab <= 1_048_576:
                raise ValueError(f"{path}: invalid vocabulary size {vocab}")
            key = (request, token)
            if key in seen:
                raise ValueError(f"{path}: duplicate logits key {key}")
            seen.add(key)
            payload = stream.read(vocab*4)
            if len(payload) != vocab*4:
                raise ValueError(f"{path}: truncated logits payload for {key}")
            values = np.frombuffer(payload, dtype="<f4").astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: nonfinite logits at {key}")
            yield key, vocab, values


def compare_logits(reference, candidate):
    import numpy as np
    rows = []
    mismatch = False
    for a, b in itertools.zip_longest(logits_records(reference, np), logits_records(candidate, np)):
        if a is None or b is None or a[:2] != b[:2]:
            mismatch = True
            # Keep consuming both files so duplicate/truncated/nonfinite tails cannot pass.
            rows.append({"reference_key": list(a[0]) if a else None,
                         "candidate_key": list(b[0]) if b else None,
                         "passed": False, "error": "different logits keys, order, count or vocabulary"})
            continue
        key, vocab, x = a
        y = b[2]
        diff = y-x
        reference_l2 = float(np.linalg.norm(x))
        error_l2 = float(np.linalg.norm(diff))
        rel = error_l2/max(reference_l2, 1e-30)
        absolute = float(np.max(np.abs(diff)))
        lx = x-float(np.max(x)); ly = y-float(np.max(y))
        lx -= math.log(float(np.exp(lx).sum())); ly -= math.log(float(np.exp(ly).sum()))
        kl = max(0.0, float(np.sum(np.exp(lx)*(lx-ly))))
        gx, gy = int(np.argmax(x)), int(np.argmax(y))
        passed = rel <= THRESHOLDS["relative_l2"] and absolute <= THRESHOLDS["max_absolute"] and kl <= THRESHOLDS["kl_reference_to_candidate"] and gx == gy
        rows.append({"key": list(key), "vocabulary": vocab, "reference_l2": reference_l2,
                     "error_l2": error_l2, "relative_l2": rel, "max_absolute": absolute,
                     "kl_reference_to_candidate": kl, "reference_greedy_id": gx,
                     "candidate_greedy_id": gy, "greedy_equal": gx == gy, "passed": passed})
    valid = [r for r in rows if "relative_l2" in r]
    return {"reference": str(reference), "candidate": str(candidate), "thresholds": THRESHOLDS,
            "scope": "All full-vocabulary records; thresholds apply independently to every record.",
            "kl_definition": "KL(softmax(reference) || softmax(candidate)), float64 stable log-softmax, natural logarithm",
            "record_count": len(rows), "keys_match": not mismatch,
            "passed": bool(rows) and not mismatch and all(r["passed"] for r in rows),
            "maximum": {k: max((r[k] for r in valid), default=None) for k in
                        ("relative_l2", "max_absolute", "kl_reference_to_candidate")},
            "all_greedy_ids_equal": bool(valid) and not mismatch and all(r["greedy_equal"] for r in valid),
            "records": rows,
            "limitation": "Matching record keys alone do not prove equal conditioning; require the paired request/input and model identity evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("reference", "candidate"):
        parser.add_argument(f"--{side}-report")
        parser.add_argument(f"--{side}-requests")
        parser.add_argument(f"--{side}-logits")
    parser.add_argument("--report", dest="candidate_report", help="alias for --candidate-report")
    parser.add_argument("--requests", dest="candidate_requests", help="alias for --candidate-requests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not any(getattr(args, f"{s}_{k}") for s in ("reference", "candidate") for k in ("report", "requests", "logits")):
        parser.error("provide at least one report, request file or logits pair")
    if bool(args.reference_logits) != bool(args.candidate_logits):
        parser.error("--reference-logits and --candidate-logits must be paired")
    result = {"format": "cf-engine-analysis", "version": 1, "thresholds": THRESHOLDS,
              "thresholds_preregistered": "2026-10-06 before candidate logits inspection; fixed in source",
              "comparisons": {}, "errors": []}
    request_rows = {}
    try:
        for side in ("reference", "candidate"):
            result[side] = {}
            path = getattr(args, f"{side}_report")
            if path:
                result[side]["report"] = summarize_report(path)
                report = result[side]["report"]
                validation = report.get("validation") or {}
                if validation.get("enabled"):
                    result["comparisons"][f"{side}_operator_validation"] = {
                        "passed": validation.get("failures") == 0 and validation.get("values", 0) > 0,
                        "reported": validation,
                        "note": "Reference outputs are used after comparison; this does not validate autoregressive CF-only outputs."}
                if side == "candidate" and report["mode"] == "cf":
                    result["comparisons"]["candidate_dispatch"] = {
                        "passed": report["cf_executed"], "cf_calls": report["cf_calls"],
                        "kernel_executed": report["kernel_executed"]}
            path = getattr(args, f"{side}_requests")
            if path:
                request_rows[side] = read_requests(path)
                result[side]["requests"] = summarize_requests(path, request_rows[side])
        if len(request_rows) == 2:
            result["comparisons"]["requests"] = compare_requests(request_rows["reference"], request_rows["candidate"])
        if "report" in result["reference"] and "report" in result["candidate"]:
            result["profile_comparison"] = compare_profiles(result["reference"]["report"], result["candidate"]["report"])
        if args.reference_logits:
            result["comparisons"]["logits"] = compare_logits(args.reference_logits, args.candidate_logits)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    comparisons = result["comparisons"]
    result["passed"] = False if result["errors"] else all(c["passed"] for c in comparisons.values()) if comparisons else None
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    for side in ("reference", "candidate"):
        report = result.get(side, {}).get("report")
        if report:
            share = report["reference_runtime_share"]
            print(f"{side}: graph={report['graph_ms']:.3f} ms, CF calls={report['cf_calls']}, reference share={share}, source-check={report['setup']['first_use_source_tensor_identity_ms']:.3f} ms")
            for phase in report["phases"]:
                groups = sorted((g for g in report["groups"] if g["phase"] == phase), key=lambda g: -g["elapsed_ms"])
                print(f"  {phase}: " + ", ".join(f"{g['category']}({g['dispatch']})={g['elapsed_ms']:.3f} ms" for g in groups[:3]))
    for name, comparison in comparisons.items():
        print(f"{name}: {'PASS' if comparison['passed'] else 'FAIL'}")
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"JSON: {output}")
    return 2 if result["errors"] else 1 if result["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
