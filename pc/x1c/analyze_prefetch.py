#!/usr/bin/env python3
"""X1c: what does ik_llama.cpp --prefetch-experts do on a pure-CPU graph?  (analysis, stdlib only)

Inputs (all produced by pc/x1c/ik_check.py in ONE memory-limited container run):
  --check   check.json of the run
  --map     results/maps/map26_ud_q3_k_m.json   (file offsets of every (layer, expert) slice)
  --instr   engine id of the INSTRUMENTED evidence build (trace file <run>/<id>/pftrace.txt)
  --strace  engine id(s) of UNMODIFIED binaries run under `strace -f -ttt -e trace=madvise`
            (<run>/<id>/strace.txt)

Trace line types written by the instrumentation (pc/x1c/instrument_prefetch.py):
  M base/size of a registered mmap     P pool start (worker threads)     E new epoch (= one graph compute)
  T full-tensor lookahead enqueue      N selective enqueue of one MoE node: caller S (scheduler hook, before
  Q jobs pushed to the queue             the graph runs) or K (kernel hook, at the MoE matmul), the ids read at
  J one worker job (skip/populate)       that moment and what enqueue_ranges did (enq: 2 = no-op, same epoch;
  C MADV_COLD pass                       4 = enqueued)

The question answered per decode step: are the expert slices the prefetcher enqueues (ids read by the first
call of the epoch) the ones the MoE kernel then uses (ids at kernel time)?
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KV = re.compile(r"(\w+)=(\S+)")


def load_regions(map_path: str):
    d = json.load(open(map_path, encoding="utf-8"))
    regs = []
    for L in d["expert_layers"]:
        for kind in ("gate_up", "down"):
            t = L[kind]
            regs.append((t["offset"], t["offset"] + t["n_bytes"], L["layer"], kind, t["nb_expert"]))
    regs.sort()
    return d, regs, [r[0] for r in regs]


def experts_in_range(regs, starts, off: int, length: int):
    """-> ({(layer, kind, expert): bytes_overlapped}, bytes outside every expert tensor)."""
    out: dict = {}
    end = off + length
    covered = 0
    i = max(0, bisect.bisect_right(starts, off) - 1)
    while i < len(regs) and regs[i][0] < end:
        s, e, layer, kind, nb = regs[i]
        lo, hi = max(off, s), min(end, e)
        if lo < hi:
            covered += hi - lo
            x0, x1 = (lo - s) // nb, (hi - 1 - s) // nb
            for x in range(x0, x1 + 1):
                a, b = max(lo, s + x * nb), min(hi, s + (x + 1) * nb)
                out[(layer, kind, x)] = out.get((layer, kind, x), 0) + (b - a)
        i += 1
    return out, length - covered


def tensor_key(name: str):
    m = re.match(r"blk\.(\d+)\.ffn_(gate_up|down|up|gate)_exps", name)
    return (int(m.group(1)), m.group(2)) if m else None


def parse_ids(s: str):
    rows = []
    for row in s.split(";"):
        rows.append([int(x) for x in row.split(",") if x not in ("", "-")])
    return rows


def parse_trace(path: Path):
    ev = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.rstrip("\n")
        if not ln or ln[1:2] != " ":
            continue
        code = ln[0]
        parts = ln.split(" ")
        try:
            t = float(parts[1])
        except (IndexError, ValueError):
            continue
        kv = dict(KV.findall(ln))
        if code == "J":
            kv["t1"] = float(parts[2])
        ev.append((code, t, kv))
    return ev


def window_of(t: float, windows):
    for name, a, b in windows:
        if a <= t <= b:
            return name
    return None


def analyze_instrumented(run: Path, eng: dict, regs, starts, map_doc) -> dict:
    eid = eng["engine_id"]
    ev = parse_trace(run / eid / "pftrace.txt")
    out: dict = {"engine_id": eid, "trace_lines": len(ev), "line_counts": dict(Counter(c for c, _, _ in ev))}
    base = None
    for c, t, kv in ev:
        if c == "M":
            base = int(kv["base"], 16)
            out["mapping"] = {"base": hex(base), "size": int(kv["size"]),
                              "size_equals_gguf": int(kv["size"]) == map_doc["file_size"]}
        if c == "P":
            out["pool"] = {"n_threads": int(kv["n_threads"]), "page": int(kv["page"])}
    # request windows
    windows = []
    for pid, pr in (eng.get("requests") or {}).items():
        for mode in ("a", "b", "stream"):
            if mode in pr:
                windows.append((f"{pid}/{mode}", pr[mode]["t_request_start_epoch"], pr[mode]["t_request_end_epoch"]))
    windows.sort(key=lambda w: w[1])
    out["request_windows"] = [w[0] for w in windows]
    # epochs
    epochs: dict[int, dict] = {}
    cur = None
    for c, t, kv in ev:
        if c == "E":
            cur = int(kv["epoch"])
            epochs[cur] = {"t": t, "N": [], "T": [], "C": [], "Q": []}
        elif c in ("N", "T", "Q") and "epoch" in kv:
            e = int(kv["epoch"])
            epochs.setdefault(e, {"t": t, "N": [], "T": [], "C": [], "Q": []})[c].append((t, kv))
        elif c == "C" and cur is not None:
            epochs[cur]["C"].append((t, kv))
    # map check: wdata - base == offset in our map
    mism = 0
    checked = 0
    name_off = {}
    for e in epochs.values():
        for t, kv in e["N"] + e["T"]:
            k = tensor_key(kv["w"])
            if k is None or base is None:
                continue
            off = int(kv["wdata"], 16) - base
            name_off[kv["w"]] = off
    for L in map_doc["expert_layers"]:
        for kind in ("gate_up", "down"):
            nm = f"blk.{L['layer']}.ffn_{kind}_exps.weight"
            if nm in name_off:
                checked += 1
                mism += int(name_off[nm] != L[kind]["offset"])
    out["tensor_offsets_vs_our_map"] = {"tensors_checked": checked, "mismatches": mism,
                                        "names_seen": len(name_off)}
    # per-epoch classification
    per_epoch = []
    for e in sorted(epochs):
        E = epochs[e]
        ncalls = E["N"]
        ntok = max((int(kv["ids_ne"].split("x")[1]) for _, kv in ncalls), default=0)
        calls = defaultdict(dict)   # tensor -> {S: kv, K: kv}
        for t, kv in ncalls:
            calls[kv["w"]].setdefault(kv["caller"], (t, kv))
        per_epoch.append({"epoch": e, "t": E["t"], "window": window_of(E["t"], windows), "n_tokens": ntok,
                          "calls": calls, "T": E["T"], "C": E["C"]})
    out["epochs_total"] = len(per_epoch)
    out["epochs_by_window"] = dict(Counter((p["window"] or "outside-requests") for p in per_epoch))
    out["epoch_n_tokens_hist"] = dict(Counter(p["n_tokens"] for p in per_epoch))

    # ---- decode steps: enqueued ids vs kernel ids ----
    def idset(kv):
        return {i for row in parse_ids(kv["ids"]) for i in row if 0 <= i < 128}

    dec = [p for p in per_epoch if p["n_tokens"] == 1 and p["window"]]
    prev_true: dict[str, set] = {}
    stats = Counter()
    overlaps = []
    src_counter = Counter()
    enq_bytes_useful = enq_bytes_wasted = need_bytes_missed = 0
    examples = []
    last_epoch_true: dict[int, dict[str, set]] = {}
    for p in per_epoch:  # ground truth per epoch (kernel-time ids), for every epoch incl. prompt batches
        last_epoch_true[p["epoch"]] = {w: idset(c["K"][1]) for w, c in p["calls"].items() if "K" in c}
    ptr_sets = Counter()
    for p in dec:
        e = p["epoch"]
        prev = last_epoch_true.get(e - 1, {})
        ptrs = set()
        for w, c in p["calls"].items():
            k = tensor_key(w)
            if k is None:
                continue
            stats["node_tensor_pairs"] += 1
            S, K = c.get("S"), c.get("K")
            if S is not None:
                stats["scheduler_hook_calls"] += 1
                stats[f"scheduler_enq_code_{S[1]['enq']}"] += 1
                ptrs.add(S[1]["ids_ptr"])
            if K is not None:
                stats["kernel_hook_calls"] += 1
                stats[f"kernel_enq_code_{K[1]['enq']}"] += 1
            if S is None or K is None:
                continue
            # the call(s) that actually pushed jobs for this tensor in this epoch (enq code 4)
            enq_calls = [c_ for c_ in (S, K) if c_[1]["enq"] == "4"]
            enq_ids = set().union(*[idset(c_[1]) for c_ in enq_calls]) if enq_calls else set()
            for c_ in enq_calls:
                stats["jobs_pushed_by_scheduler_hook" if c_ is S else "jobs_pushed_by_kernel_hook"] += 1
            if not enq_calls:
                stats["no_jobs_pushed_for_tensor"] += 1
            first = enq_calls[0] if enq_calls else S
            true_ids = idset(K[1])
            stats["pairs_with_both"] += 1
            stats["enqueued_equals_true"] += int(enq_ids == true_ids)
            ov = len(enq_ids & true_ids)
            overlaps.append(ov)
            nb = int(first[1]["stride"])
            enq_bytes_useful += ov * nb
            enq_bytes_wasted += len(enq_ids - true_ids) * nb
            need_bytes_missed += len(true_ids - enq_ids) * nb
            if enq_ids and enq_ids != true_ids:
                # where do the enqueued ids come from? look for the same set among the previous epoch's true ids
                if prev.get(w) == enq_ids:
                    src_counter["previous epoch, same tensor"] += 1
                else:
                    hit = [w2 for w2, s2 in prev.items() if s2 == enq_ids]
                    if hit:
                        src_counter["previous epoch, other tensor: " + ",".join(sorted({str(tensor_key(h)) for h in hit}))[:80]] += 1
                    else:
                        cur_hit = [w2 for w2, s2 in last_epoch_true.get(e, {}).items() if s2 == enq_ids and w2 != w]
                        src_counter["same epoch, other tensor" if cur_hit else "no matching id set found"] += 1
                if len(examples) < 6:
                    examples.append({"epoch": e, "window": p["window"], "tensor": w,
                                     "ids_read_when_enqueued": sorted(enq_ids), "ids_used_by_kernel": sorted(true_ids),
                                     "previous_epoch_true_ids_same_tensor": sorted(prev.get(w, []))})
        ptr_sets[len(ptrs)] += 1
    n = max(1, stats["pairs_with_both"])
    out["decode_steps"] = {
        "decode_epochs_inside_requests": len(dec),
        "counts": dict(stats),
        "fraction_enqueued_set_equals_kernel_set": round(stats["enqueued_equals_true"] / n, 4),
        "mean_overlap_of_8": round(statistics.fmean(overlaps), 3) if overlaps else None,
        "overlap_histogram": dict(sorted(Counter(overlaps).items())),
        "enqueued_bytes_useful": enq_bytes_useful, "enqueued_bytes_wasted": enq_bytes_wasted,
        "needed_bytes_not_enqueued": need_bytes_missed,
        "wasted_fraction_of_enqueued": round(enq_bytes_wasted / max(1, enq_bytes_useful + enq_bytes_wasted), 4),
        "source_of_mismatching_id_sets": dict(src_counter),
        "distinct_ids_buffers_per_epoch_hist": dict(ptr_sets),
        "examples": examples,
    }

    # ---- prompt batches ----
    pb = [p for p in per_epoch if p["n_tokens"] > 1]
    rows = []
    for p in pb:
        tb = sum(int(kv["bytes"]) for _, kv in p["T"] if kv["enq"] == "4")
        tj = sum(int(kv["jobs"]) for _, kv in p["T"] if kv["enq"] == "4")
        s_codes = Counter(c["S"][1]["enq"] for c in p["calls"].values() if "S" in c)
        k_codes = Counter(c["K"][1]["enq"] for c in p["calls"].values() if "K" in c)
        # selective enqueue quality for batches without lookahead
        enq_e = true_e = both = 0
        for w, c in p["calls"].items():
            if "S" in c and "K" in c:
                ec = [c_ for c_ in (c["S"], c["K"]) if c_[1]["enq"] == "4"]
                a = set().union(*[idset(c_[1]) for c_ in ec]) if ec else set()
                b = idset(c["K"][1])
                enq_e += len(a)
                true_e += len(b)
                both += len(a & b)
        rows.append({"epoch": p["epoch"], "window": p["window"] or "outside-requests", "n_tokens": p["n_tokens"],
                     "lookahead_full_tensors_enqueued": sum(1 for _, kv in p["T"] if kv["enq"] == "4"),
                     "lookahead_bytes_enqueued": tb, "lookahead_jobs": tj,
                     "scheduler_selective_enq_codes": dict(s_codes), "kernel_selective_enq_codes": dict(k_codes),
                     "selective_expert_slices_enqueued": enq_e, "expert_slices_used": true_e,
                     "enqueued_and_used": both,
                     "madv_cold_passes": len(p["C"]),
                     "madv_cold_chunks": sum(int(kv["chunks"]) for _, kv in p["C"]),
                     "madv_cold_bytes": sum(int(kv["bytes"]) for _, kv in p["C"])})
    out["prompt_batches"] = rows

    # ---- worker jobs ----
    jobs = [(t, kv) for c, t, kv in ev if c == "J"]
    by_win = defaultdict(lambda: Counter())
    useful = wasted = nonexp = 0
    for t, kv in jobs:
        w = window_of(t, windows) or "outside-requests"
        res = kv["res"]
        ln = int(kv["len"])
        by_win[w][f"jobs_res{res}"] += 1
        by_win[w][f"bytes_res{res}"] += ln
        by_win[w]["madvise_seconds_x1e6"] += int((kv["t1"] - t) * 1e6) if res != "0" else 0
        if res == "1" and base is not None:
            ex, outside = experts_in_range(regs, starts, int(kv["addr"], 16) - base, ln)
            je = int(kv["epoch"])
            truth = last_epoch_true.get(je, {})
            for (layer, kind, x), b in ex.items():
                tset = truth.get(f"blk.{layer}.ffn_{kind}_exps.weight")
                if tset is not None and x in tset:
                    useful += b
                else:
                    wasted += b
            nonexp += outside
    out["worker_jobs"] = {
        "legend": "res0 = skipped (all pages already resident), res1 = MADV_POPULATE_READ ok, res2 = madvise failed",
        "total": len(jobs),
        "by_request_window": {w: dict(c) for w, c in by_win.items()},
        "populated_bytes_in_slices_the_kernel_used_in_that_epoch": useful,
        "populated_bytes_in_slices_not_used_in_that_epoch": wasted,
        "populated_bytes_outside_expert_tensors": nonexp,
        "note": "a job's epoch is the epoch stored in its ticket when the worker ran it (the ticket is re-stamped "
                "when the tensor is enqueued again), so for lookahead sweeps that are still draining after their "
                "graph finished this attribution is approximate; the decode-step table above is exact",
    }
    if jobs:
        out["worker_jobs"]["first_job_t"] = jobs[0][0]
        out["worker_jobs"]["last_job_t"] = jobs[-1][0]
    out["_truth"] = {str(e): {w: sorted(s) for w, s in d.items()} for e, d in last_epoch_true.items()}
    out["_epoch_meta"] = [{"epoch": p["epoch"], "t": p["t"], "window": p["window"], "n_tokens": p["n_tokens"]}
                          for p in per_epoch]
    return out


MADV = re.compile(r"^(\d+)\s+(\d+\.\d+)\s+madvise\((0x[0-9a-f]+),\s*(\d+),\s*([A-Z_0-9x|a-f /*?]+?)(?:\)\s*=\s*(-?\d+).*|\s*<unfinished \.\.\.>)$")


def analyze_strace(run: Path, eng: dict, regs, starts, truth_by_request: dict | None) -> dict:
    eid = eng["engine_id"]
    p = run / eid / "strace.txt"
    out: dict = {"engine_id": eid}
    if not p.exists():
        out["error"] = "no strace.txt"
        return out
    base = None
    mm = (eng.get("snapshots", {}).get("after_load", {}) or {}).get("model_mappings") or []
    if mm:
        base = int(mm[0]["start"], 16) - int(mm[0]["file_offset"])
    out["model_mapping_base"] = hex(base) if base is not None else None
    windows = []
    for pid, pr in (eng.get("requests") or {}).items():
        for mode in ("a", "b", "stream"):
            if mode in pr:
                windows.append((f"{pid}/{mode}", pr[mode]["t_request_start_epoch"], pr[mode]["t_request_end_epoch"]))
    adv = Counter()
    tids = defaultdict(set)
    per_win = defaultdict(lambda: {"calls": 0, "bytes": 0, "experts": {}, "non_expert_bytes": 0})
    n_lines = 0
    for ln in open(p, encoding="utf-8", errors="replace"):
        n_lines += 1
        m = MADV.match(ln.rstrip("\n"))
        if not m:
            continue
        tid, ts, addr, length, advice = int(m.group(1)), float(m.group(2)), int(m.group(3), 16), int(m.group(4)), m.group(5).strip()
        advice = advice.split()[0]
        adv[advice] += 1
        tids[advice].add(tid)
        if advice not in ("MADV_POPULATE_READ", "0x16"):
            continue
        w = window_of(ts, windows) or ("load" if ts < eng.get("t_ready_epoch", 0) else "outside-requests")
        pw = per_win[w]
        pw["calls"] += 1
        pw["bytes"] += length
        if base is not None:
            ex, outside = experts_in_range(regs, starts, addr - base, length)
            for k, b in ex.items():
                pw["experts"][k] = pw["experts"].get(k, 0) + b
            pw["non_expert_bytes"] += outside
    out["strace_lines"] = n_lines
    out["madvise_calls_by_advice"] = dict(adv)
    out["threads_issuing"] = {a: len(t) for a, t in tids.items()}
    rows = {}
    for w, pw in per_win.items():
        row = {"populate_calls": pw["calls"], "populate_bytes": pw["bytes"],
               "distinct_expert_slices_touched": len(pw["experts"]), "non_expert_bytes": pw["non_expert_bytes"],
               "layers_touched": len({k[0] for k in pw["experts"]})}
        if truth_by_request and w in truth_by_request:
            used = truth_by_request[w]
            pe = set(pw["experts"])
            row["slices_populated_and_used_in_this_request"] = len(pe & used)
            row["slices_populated_but_never_used_in_this_request"] = len(pe - used)
            row["slices_used_in_this_request"] = len(used)
            row["bytes_populated_in_never_used_slices"] = sum(b for k, b in pw["experts"].items() if k not in used)
        rows[w] = row
    out["populate_by_window"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--strace", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.check).parent
    chk = json.load(open(a.check, encoding="utf-8"))
    engs = {e["engine_id"]: e for e in chk["engines"]}
    map_doc, regs, starts = load_regions(a.map)
    res: dict = {"schema": "pigemma-x1c-prefetch-analysis v1", "check": str(a.check), "map": a.map,
                 "kind": chk.get("kind"), "cgroup_limit_bytes": (chk.get("cgroup_at_start") or {}).get("memory.limit_in_bytes")}
    ins = analyze_instrumented(run, engs[a.instr], regs, starts, map_doc)
    truth = ins.pop("_truth")
    meta = ins.pop("_epoch_meta")
    res["instrumented"] = ins
    # ground truth per request window: every slice any epoch of that request used
    tbr: dict[str, set] = defaultdict(set)
    for m in meta:
        if not m["window"]:
            continue
        for w, ids in truth.get(str(m["epoch"]), {}).items():
            k = tensor_key(w)
            if k:
                for x in ids:
                    tbr[m["window"]].add((k[0], k[1], x))
    res["ground_truth_slices_used_per_request"] = {w: len(s) for w, s in tbr.items()}
    # tokens identical between the instrumented and the unmodified runs?
    res["strace"] = []
    for sid in [s for s in a.strace.split(",") if s]:
        e = engs[sid]
        same = {}
        for pid, pr in (e.get("requests") or {}).items():
            ref = (engs[a.instr].get("requests") or {}).get(pid, {})
            for mode in ("a", "stream"):
                if mode in pr and mode in ref:
                    same[f"{pid}/{mode}"] = (pr[mode].get("text") == ref[mode].get("text"))
        r = analyze_strace(run, e, regs, starts, tbr)
        r["generated_text_identical_to_instrumented_run"] = same
        res["strace"].append(r)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: res["instrumented"].get(k) for k in ("pool", "mapping", "tensor_offsets_vs_our_map", "epochs_by_window",
                                                               "epoch_n_tokens_hist")}, indent=1))
    d = res["instrumented"]["decode_steps"]
    print(json.dumps({k: d[k] for k in d if k != "examples"}, indent=1))
    print(json.dumps(res["instrumented"]["prompt_batches"], indent=1)[:3000])
    for r in res["strace"]:
        print(json.dumps({k: r.get(k) for k in ("engine_id", "madvise_calls_by_advice", "threads_issuing", "populate_by_window",
                                                "generated_text_identical_to_instrumented_run")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
