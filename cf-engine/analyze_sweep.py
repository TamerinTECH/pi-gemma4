"""Offline analysis of a single-process, explicitly alternating CF experiment.

python cf-engine/analyze_sweep.py --directory RUN --out comparison.json
python cf-engine/analyze_sweep.py --directory RUN --pairs quality-pairs.json --out quality.json

RUN contains profile.json, requests.jsonl, and optionally prompts.json. Quality
requires logits.bin and routes.bin. No model, Pi job or inference is launched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import itertools
import json
import math
import mmap
from pathlib import Path
import statistics
import struct
import sys

_spec = importlib.util.spec_from_file_location("cf_existing_analysis", Path(__file__).with_name("analyze.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
THRESHOLDS = dict(base.THRESHOLDS)
IDENTITY_FIELDS = ("n_prompt_tokens", "n_keep", "n_prefill", "max_tokens", "server_stop", "prefill_calls")
EXECUTION_NUMBERS = ("cf_calls", "reference_calls", "cf_ms", "reference_ms", "prepare_ms", "multiply_ms",
                     "source_check_ms", "layer0_complete_ffn_prefill_ms", "layer0_complete_ffn_decode_ms",
                     "storage_read_bytes", "major_faults")


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summary(values):
    values = [value for value in values if value is not None]
    return {"count": len(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "stdev": statistics.stdev(values) if len(values) > 1 else None}


def ratio(numerator, denominator, factor=1.0):
    return factor*numerator/denominator if denominator > 0 else None


def sampled_intertoken_timing(times):
    """Consecutive model-sampling gaps, including hidden/control/EOS samples.

    These timestamps do not map tokens to visible answer pieces. The first
    sample and the post-final-sample tail are not intertoken intervals.
    """
    times = [base.number(value, "sampled_ms") for value in times]
    if times != sorted(times):
        raise ValueError("sampled_ms must be monotonic")
    gaps = [end-start for start, end in zip(times, times[1:])]
    longest = None
    if gaps:
        index = max(range(len(gaps)), key=gaps.__getitem__)
        longest = {"duration_ms": gaps[index], "from_sample_index": index,
                   "to_sample_index": index+1, "start_request_ms": times[index],
                   "end_request_ms": times[index+1]}
    return {"sampled_intertoken_interval_count": len(gaps),
            "sampled_intertoken_median_ms": statistics.median(gaps) if gaps else None,
            "sampled_intertoken_p95_ms": base.percentile(gaps, 0.95),
            "sampled_intertoken_max_ms": max(gaps) if gaps else None,
            "longest_sampled_token_stall": longest}


def input_identity(prompt):
    if not isinstance(prompt, dict):
        raise ValueError("prompt entry must be an object")
    if "tokens" in prompt:
        tokens = prompt["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("input tokens must be a nonempty list")
        for token in tokens:
            integer(token, "input token")
        payload = {"tokens": tokens}
    elif isinstance(prompt.get("prompt"), str):
        payload = {"prompt": prompt["prompt"]}
    else:
        raise ValueError("prompt entry requires prompt text or exact tokens")
    # Prefix retention affects conditioning even when prompt text is identical.
    payload["keep"] = prompt.get("keep", 0)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_request_pair(left, right):
    differences = [key for key in IDENTITY_FIELDS if left.get(key) != right.get(key)]
    exact = left["gen_tokens"] == right["gen_tokens"]
    text = left["text"] == right["text"]
    stop = left["stop"] == right["stop"]
    first = next((i for i, (a, b) in enumerate(itertools.zip_longest(
        left["gen_tokens"], right["gen_tokens"])) if a != b), None)
    return {"passed": bool(left["gen_tokens"]) and not differences and exact and text and stop,
            "workload_field_differences": differences, "token_ids_exact": exact, "text_exact": text,
            "stop_equal": stop, "tokens_compared": len(left["gen_tokens"]),
            "first_mismatching_token_index": first}


def join_requests(profile, requests, prompts=None):
    execution = profile.get("request_execution")
    if not isinstance(execution, list) or not execution:
        raise ValueError("profile has no per-request execution records")
    by_index = {}
    for event in execution:
        index = integer(event["request"], "execution request")
        if index in by_index:
            raise ValueError(f"duplicate execution request {index}")
        for field in EXECUTION_NUMBERS:
            base.number(event[field], f"request {index} {field}")
        if not isinstance(event.get("kernel"), str) or not event["kernel"]:
            raise ValueError(f"request {index}: missing kernel identity")
        by_index[index] = event
    seen = set()
    rows = []
    for request in requests:
        index = integer(request["chunk_id"], "request chunk_id")
        if index in seen or index not in by_index:
            raise ValueError(f"duplicate request or missing execution record {index}")
        seen.add(index)
        event = by_index[index]
        for field in ("n_sampled", "n_gen_evaluated", "n_prefill", "n_keep", "n_prompt_tokens"):
            integer(request[field], f"request {index} {field}")
        tgen = request["t_total_ms"] - request["t_prefill_ms"]
        if tgen < 0:
            raise ValueError(f"request {index}: negative generation interval")
        times = request["sampled_ms"]
        interval = times[-1] - times[0] if len(times) > 1 else 0
        actual = ("cf-portable" if event["kernel"] == "dotprod" and not profile.get("dotprod_available")
                  else event["kernel"] if event["kernel"] == "reference" else "cf-"+event["kernel"])
        row = {**request, "request": index, "warmup_excluded": index == 0, "kernel": event["kernel"],
               "arithmetic_executed": actual, "execution": event,
               "dispatch_verified": event["cf_calls"] == 0 if event["kernel"] == "reference" else event["cf_calls"] > 0,
               "t_gen_ms": tgen,
               "decode_evaluated_tokens_per_s": ratio(request["n_gen_evaluated"], tgen, 1000),
               "decode_sample_interval_tokens_per_s": ratio(max(0, len(times)-1), interval, 1000),
               "prefill_tokens_per_s": ratio(request["n_prefill"], request["t_prefill_ms"], 1000),
               "full_request_sampled_model_tokens_per_s": ratio(request["n_sampled"], request["t_total_ms"], 1000),
               "sampled_tokens_including_terminal_eos": request["n_sampled"],
               "sampled_tokens_excluding_terminal_eos": request["n_sampled"] - (1 if request["stop"] == "eos" else 0),
               "first_sample_ms": times[0] if times else None,
               "first_visible_answer_ms": None, "visible_answer_tokens_per_s": None,
               "visible_answer_status": "unknown: no per-piece visible-channel timestamp mapping",
               **sampled_intertoken_timing(times),
               "reference_runtime_share": ratio(event["reference_ms"], event["reference_ms"]+event["cf_ms"]),
               "layer0_complete_ffn_decode_ms_per_evaluated_token": ratio(
                   event["layer0_complete_ffn_decode_ms"], request["n_gen_evaluated"])}
        if prompts is not None:
            if index >= len(prompts) or prompts[index].get("id") != request["id"]:
                raise ValueError(f"request {index}: prompt file id/order mismatch")
            row["input_identity"] = input_identity(prompts[index])
        rows.append(row)
    if seen != set(by_index):
        raise ValueError("profile and request JSONL contain different request sets")
    if prompts is not None and len(prompts) != len(rows):
        raise ValueError("prompt file and completed request counts differ")
    return sorted(rows, key=lambda row: row["request"])


def summarize_performance(rows, has_prompts):
    eligible = [row for row in rows if not row["warmup_excluded"]]
    groups = defaultdict(list)
    for row in eligible:
        groups[row["kernel"]].append(row)
    metrics = ("t_prefill_ms", "t_gen_ms", "t_total_ms", "first_sample_ms", "n_sampled", "n_gen_evaluated",
               "decode_evaluated_tokens_per_s", "decode_sample_interval_tokens_per_s", "prefill_tokens_per_s",
               "full_request_sampled_model_tokens_per_s", "reference_runtime_share",
               "sampled_intertoken_median_ms", "sampled_intertoken_p95_ms", "sampled_intertoken_max_ms",
               "layer0_complete_ffn_decode_ms_per_evaluated_token")
    output = {}
    for kernel, members in groups.items():
        output[kernel] = {"request_indices": [row["request"] for row in members],
            "count": len(members), "dispatch_verified": all(row["dispatch_verified"] for row in members),
            "metrics": {field: summary(row[field] for row in members) for field in metrics},
            "execution_metrics": {field: summary(row["execution"][field] for row in members) for field in EXECUTION_NUMBERS},
            "totals": {"sampled_model_tokens_including_terminal_eos": sum(row["n_sampled"] for row in members),
                "sampled_model_tokens_excluding_terminal_eos": sum(row["sampled_tokens_excluding_terminal_eos"] for row in members),
                "evaluated_generation_tokens": sum(row["n_gen_evaluated"] for row in members),
                "request_ms": sum(row["t_total_ms"] for row in members),
                "storage_read_bytes": sum(row["execution"]["storage_read_bytes"] for row in members)},
            "all_requests_zero_storage_reads": all(row["execution"]["storage_read_bytes"] == 0 for row in members)}
    prompt_groups = defaultdict(list)
    for row in eligible:
        prompt_groups[row.get("input_identity", "same-prompt-experiment-assertion")].append(row)
    matches = []
    for members in prompt_groups.values():
        if len(members) < 2:
            continue
        anchor = members[0]
        for row in members[1:]:
            matches.append({"reference_request": anchor["request"], "candidate_request": row["request"],
                            **compare_request_pair(anchor, row)})
    return {"warmup_excluded_request": 0, "groups": output,
            "order": [{"request": row["request"], "kernel": row["kernel"]} for row in rows],
            "same_prompt_output_check": {"input_identity_status": "exact prompt/tokens and keep matched within one fixed-model process" if has_prompts else "same prompt asserted by experiment; not proven from request counts",
                "distinct_input_groups": len(prompt_groups) if has_prompts else None,
                "all_nonwarmup_inputs_identical": len(prompt_groups) == 1 if has_prompts else None,
                "passed": bool(matches) and all(row["passed"] for row in matches), "comparisons": matches},
            "notes": ["Per-kernel summaries exclude request 0 only; per-request records retain it.",
                "Decode evaluated-token rate is n_gen_evaluated / (t_total_ms - t_prefill_ms).",
                "Sample-interval rate is (n_sampled - 1) / (last_sample_ms - first_sample_ms).",
                "n_sampled includes EOS/control/possibly hidden reasoning; excluding EOS does not imply visible or useful tokens.",
                "First-sample latency is a model-token proxy, never a first-visible-answer measurement.",
                "Sampled intertoken gaps are consecutive sampled_ms differences, including hidden/control/terminal-EOS samples; the first-sample and final-sample-to-request-end intervals are excluded.",
                "Per-request intertoken median is the ordinary sample median; p95 uses nearest rank ceil(0.95*n). Longest stall is the maximum gap, with zero-based sample indices (earliest occurrence if tied). Fewer than two samples yields no interval statistics.",
                "Grouped intertoken metrics summarize per-request statistics, not a pooled-token latency percentile; no visible-answer latency or rate is inferred.",
                "Grouped quality workloads may differ; pooled per-kernel summaries are descriptive unless inputs match.",
                "Zero storage reads do not measure DRAM or cache traffic.",
                "Complete layer-0 FFN time includes shared FFN, expert FFN, norms, router, combination, residual and layer scale.",
                "Adapter initialization/model loading precede these requests; first-use source checks are separately retained in request execution."]}


class BinaryRecords:
    """Read-only mmap and bounded indexes; no complete logit dump is copied."""
    def __init__(self, path, kind):
        import numpy as np
        self.path = Path(path)
        self.stream = self.path.open("rb")
        self.mapping = None
        self.records = defaultdict(list)
        try:
            length = self.path.stat().st_size
            if not length:
                raise ValueError(f"{path}: empty {kind} dump")
            self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
            position = 0
            seen = set()
            while position < length:
                header_bytes = 24 if kind == "logits" else 32
                if length-position < header_bytes:
                    raise ValueError(f"{path}: truncated {kind} header")
                if kind == "logits":
                    vocab, request, token = struct.unpack_from("<QQQ", self.mapping, position)
                    position += 24
                    if not 1 <= vocab <= 1_048_576 or vocab*4 > length-position:
                        raise ValueError(f"{path}: invalid/truncated logits payload")
                    if (request, token) in seen:
                        raise ValueError(f"{path}: duplicate logit request/token")
                    seen.add((request, token))
                    if not np.isfinite(np.frombuffer(self.mapping, dtype="<f4", count=vocab, offset=position)).all():
                        raise ValueError(f"{path}: nonfinite logits at request {request}, token {token}")
                    self.records[request].append({"token": token, "vocab": vocab, "offset": position})
                    position += vocab*4
                else:
                    name_length, slots, tokens, request = struct.unpack_from("<QQQQ", self.mapping, position)
                    position += 32
                    count = slots*tokens
                    if not 1 <= name_length <= 4096 or not 1 <= slots <= 128 or not 1 <= tokens <= 1_048_576:
                        raise ValueError(f"{path}: invalid route dimensions")
                    if count > 16_777_216 or name_length+count*4 > length-position:
                        raise ValueError(f"{path}: truncated/oversized route payload")
                    name = self.mapping[position:position+name_length].decode("utf-8")
                    position += name_length
                    if np.any(np.frombuffer(self.mapping, dtype="<i4", count=count, offset=position) < 0):
                        raise ValueError(f"{path}: negative expert ID")
                    self.records[request].append({"name": name, "slots": slots, "tokens": tokens, "offset": position})
                    position += count*4
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def compare_pair_logits(index, reference, candidate, expected_reference, expected_candidate):
    import numpy as np
    left, right = index.records.get(reference, []), index.records.get(candidate, [])
    keys_match = bool(left) and bool(right) and len(left) == len(right)
    coverage = ([row["token"] for row in left] == list(range(expected_reference)) and
                [row["token"] for row in right] == list(range(expected_candidate)) and
                expected_reference > 0 and expected_candidate > 0)
    rows = []
    for a, b in itertools.zip_longest(left, right):
        if a is None or b is None or (a["token"], a["vocab"]) != (b["token"], b["vocab"]):
            keys_match = False
            rows.append({"passed": False, "error": "different logit token order/count/vocabulary",
                         "reference_token": a["token"] if a else None, "candidate_token": b["token"] if b else None})
            continue
        # Same fixed formulas and thresholds as analyze.compare_logits, with
        # request IDs paired explicitly rather than creating remapped files.
        x = np.frombuffer(index.mapping, dtype="<f4", count=a["vocab"], offset=a["offset"]).astype(np.float64)
        y = np.frombuffer(index.mapping, dtype="<f4", count=b["vocab"], offset=b["offset"]).astype(np.float64)
        diff = y-x
        relative = float(np.linalg.norm(diff))/max(float(np.linalg.norm(x)), 1e-30)
        absolute = float(np.max(np.abs(diff)))
        lx, ly = x-float(np.max(x)), y-float(np.max(y))
        lx -= math.log(float(np.exp(lx).sum())); ly -= math.log(float(np.exp(ly).sum()))
        kl = max(0.0, float(np.sum(np.exp(lx)*(lx-ly))))
        gx, gy = int(np.argmax(x)), int(np.argmax(y))
        passed = (relative <= THRESHOLDS["relative_l2"] and absolute <= THRESHOLDS["max_absolute"]
                  and kl <= THRESHOLDS["kl_reference_to_candidate"] and gx == gy)
        rows.append({"token": a["token"], "vocabulary": a["vocab"], "relative_l2": relative,
                     "max_absolute": absolute, "kl_reference_to_candidate": kl,
                     "reference_greedy_id": gx, "candidate_greedy_id": gy, "greedy_equal": gx == gy, "passed": passed})
    numeric = [row for row in rows if "relative_l2" in row]
    return {"passed": keys_match and coverage and bool(rows) and all(row["passed"] for row in rows),
            "keys_match": keys_match, "all_sampled_tokens_have_logits": coverage,
            "reference_records": len(left), "candidate_records": len(right),
            "all_greedy_ids_equal": bool(numeric) and keys_match and all(row["greedy_equal"] for row in numeric),
            "maximum": {key: max((row[key] for row in numeric), default=None)
                        for key in ("relative_l2", "max_absolute", "kl_reference_to_candidate")},
            "records": rows}


def compare_pair_routes(index, reference, candidate, expected_names):
    import numpy as np
    left, right = index.records.get(reference, []), index.records.get(candidate, [])
    coverage = bool(left) and bool(right)
    left_names, right_names = {row["name"] for row in left}, {row["name"] for row in right}
    if expected_names:
        coverage = coverage and expected_names <= left_names and expected_names <= right_names
    mismatches, ids_compared = [], 0
    for ordinal, (a, b) in enumerate(itertools.zip_longest(left, right)):
        if a is None or b is None or any(a[key] != b[key] for key in ("name", "slots", "tokens")):
            mismatches.append({"record": ordinal, "reason": "node order/name/shape/count mismatch"})
            continue
        count = a["slots"]*a["tokens"]
        x = np.frombuffer(index.mapping, dtype="<i4", count=count, offset=a["offset"])
        y = np.frombuffer(index.mapping, dtype="<i4", count=count, offset=b["offset"])
        unequal = np.flatnonzero(x != y)
        ids_compared += count
        if unequal.size:
            first = int(unequal[0])
            mismatches.append({"record": ordinal, "name": a["name"], "reason": "expert IDs differ",
                "mismatching_ids": int(unequal.size), "first_token": first//a["slots"],
                "first_slot": first % a["slots"], "reference_id": int(x[first]), "candidate_id": int(y[first])})
        del x, y
    return {"passed": coverage and not mismatches, "requested_route_records_present": coverage,
            "reference_records": len(left), "candidate_records": len(right), "expert_ids_compared": ids_compared,
            "expected_projection_names": sorted(expected_names),
            "reference_projection_names": sorted(left_names), "candidate_projection_names": sorted(right_names),
            "mismatches": mismatches}


def analyze_quality(directory, pairs, rows, profile, has_prompts):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("quality pairs must be a nonempty list")
    requests = {row["request"]: row for row in rows}
    expected_names = {row["weight_name"] for row in profile.get("ops", [])
                      if row.get("op") == "MUL_MAT_ID" and row.get("weight_name")}
    expected_route_counts = defaultdict(int)
    for event in profile.get("ops", []):
        if event.get("op") == "MUL_MAT_ID" and event.get("weight_name"):
            expected_route_counts[event["weight_name"]] += integer(event["calls"], "profile projection calls")
    checks, seen = [], set()
    with BinaryRecords(directory/"logits.bin", "logits") as logits, BinaryRecords(directory/"routes.bin", "routes") as routes:
        if (set(logits.records) | set(routes.records)) - set(requests):
            raise ValueError("binary dump contains request IDs absent from completed request records")
        actual_route_counts = defaultdict(int)
        for records in routes.records.values():
            for record in records:
                actual_route_counts[record["name"]] += 1
        # Profile begin writes one route record and profile end counts that same
        # leading MUL_MAT_ID node. This catches whole-record truncation that a
        # binary bounds check alone cannot detect, including matching omissions.
        route_count_coverage = bool(expected_route_counts) and dict(expected_route_counts) == dict(actual_route_counts)
        for pair in pairs:
            name = pair["id"]
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("empty or duplicate quality pair ID")
            seen.add(name)
            a, b = integer(pair["reference"], "reference request"), integer(pair["candidate"], "candidate request")
            item = {"id": name, "reference": a, "candidate": b}
            if a == b or a not in requests or b not in requests:
                item.update(passed=False, error="distinct complete request records absent")
                checks.append(item)
                continue
            left, right = requests[a], requests[b]
            item["requests"] = compare_request_pair(left, right)
            item["input_identity_verified"] = has_prompts and left["input_identity"] == right["input_identity"]
            item["backend_identity_verified"] = (left["kernel"] == "reference" and right["kernel"] != "reference"
                and left["dispatch_verified"] and right["dispatch_verified"])
            item["logits"] = compare_pair_logits(logits, a, b, left["n_sampled"], right["n_sampled"])
            item["routes"] = compare_pair_routes(routes, a, b, expected_names)
            item["comparisons_passed"] = all(item[key]["passed"] for key in ("requests", "logits", "routes"))
            item["passed"] = item["comparisons_passed"] and item["input_identity_verified"] and item["backend_identity_verified"]
            checks.append(item)
    diagnostic = bool(profile.get("validation", {}).get("enabled"))
    return {"passed": bool(checks) and len(checks) == len(pairs) and not diagnostic and route_count_coverage and all(row["passed"] for row in checks),
            "requested_pairs": len(pairs), "reported_pairs": len(checks), "pairs": checks,
            "route_dump_matches_profile_call_counts": route_count_coverage,
            "expected_route_records_by_projection": dict(expected_route_counts),
            "actual_route_records_by_projection": dict(actual_route_counts),
            "input_identity_status": "provided exact input file" if has_prompts else "asserted, not proven; full quality gate cannot pass",
            "operator_diagnostic_mode": diagnostic, "thresholds": THRESHOLDS,
            "threshold_scope": "independently applied to every full-vocabulary sampled-token record",
            "kl_definition": "KL(softmax(reference) || softmax(candidate)), float64 stable log-softmax, natural logarithm",
            "limitations": ["Same-process model/tokenizer/global configuration are fixed; request backend identity and inputs are checked separately.",
                "A pass establishes paired numerical/routing/output agreement on these requests, not general task quality or distribution equivalence.",
                "Ordered routing records include every captured projection node; projection-name completeness is checked when profile weight names exist.",
                "Reference-output diagnostic mode cannot establish full candidate autoregressive correctness."]}


def analyze_directory(directory, pairs=None, prompts_path=None):
    directory = Path(directory)
    profile = load_json(directory/"profile.json")
    if profile.get("format") != "cf-engine-run" or profile.get("version") != 1:
        raise ValueError("unsupported profile format/version")
    requests = base.read_requests(directory/"requests.jsonl")
    prompt_file = Path(prompts_path) if prompts_path else directory/"prompts.json"
    prompts = load_json(prompt_file) if prompt_file.is_file() else None
    if prompts_path is not None and prompts is None:
        raise ValueError(f"specified prompts file not found: {prompt_file}")
    if prompts is not None and not isinstance(prompts, list):
        raise ValueError("prompts file must contain a list")
    rows = join_requests(profile, requests, prompts)
    result = {"format": "cf-engine-sweep-analysis", "version": 1, "directory": str(directory),
              "thresholds": THRESHOLDS, "source_profile": str(directory/"profile.json"),
              "prompts": str(prompt_file) if prompts is not None else None,
              "execution_contract": profile.get("execution_contract"),
              "startup": {key: profile.get(key) for key in ("init_and_identity_check_ms", "first_use_source_tensor_identity_ms")},
              "memory": {key: profile.get(key) for key in ("memory_policy", "mapped_packed_bytes", "source_file_bytes", "scratch_allocated_bytes", "scratch_active_peak_bytes", "peak_rss_kib")},
              "performance": summarize_performance(rows, prompts is not None),
              "requests": [{key: value for key, value in row.items() if key != "input_identity"} for row in rows]}
    if pairs is not None:
        result["quality"] = analyze_quality(directory, pairs, rows, profile, prompts is not None)
        result["passed"] = result["quality"]["passed"]
    else:
        result["passed"] = (bool(rows) and all(row["dispatch_verified"] for row in rows)
                            and result["performance"]["same_prompt_output_check"]["passed"])
        result["pass_scope"] = "request/backend/output consistency only; no logits/routing/general-quality claim"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="JSON list of {id,reference,candidate} request-index pairs")
    parser.add_argument("--prompts", type=Path, help="exact input JSON; defaults to DIRECTORY/prompts.json when present")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {"format": "cf-engine-sweep-analysis", "version": 1, "passed": False, "errors": []}
    try:
        result.update(analyze_directory(args.directory, load_json(args.pairs) if args.pairs else None, args.prompts))
    except (OSError, ValueError, TypeError, KeyError, ImportError, struct.error) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["passed"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "errors": result["errors"], "report": str(args.out)}))
    return 2 if result["errors"] else 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
