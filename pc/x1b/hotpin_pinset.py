#!/usr/bin/env python3
"""X1b: HotPin pin sets for OUR exact GGUF, produced with HotPin's own tool, plus page accounting.

Sub-commands (host Python or the container; stdlib only, the HotPin tool itself wants `gguf`):

  original  --routing R --budget-gb B --out F     run third_party/hotpin-llm/tools/make_hot_file.py UNMODIFIED
                                                  (its own slice estimate file_size/(n_layer*n_expert))
  order     --routing R --out F                   the tool's FULL ranking order (same tool, huge budget)
  match     --order F --out G (--expert-pages N | --total-pages N | --plan C.plan.json [--mode expert|total])
                                                  LOCALLY ADAPTED budget: strict prefix of HotPin's own order whose
                                                  predicted unique locked pages (HotPin's real mlock ranges) fit
  predict   --hot F                               predicted lock accounting of any hot file

Lock model (read from third_party/hotpin-llm/patches/hotpin_l1_pin.diff + llama_mlock at a6647b1):
for every "layer expert" line and every existing blk.L.ffn_{gate,up,down,gate_up}_exps.weight tensor,
mlock(t->data + e*nb[2], roundup(nb[2], page)).  mlock(2) itself extends the start DOWN to a page
boundary, so the locked pages are [floor(off/P), ceil((off + roundup(nb,P))/P)) in file-offset space
(the model mmap starts at file offset 0, so address and file-offset page alignment coincide; valid
only with -nr, when t->data points into the mapping).  No HotPin code is copied here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOTPIN_TOOL = REPO / "third_party" / "hotpin-llm" / "tools" / "make_hot_file.py"
MAP_DEFAULT = REPO / "results" / "maps" / "map26_ud_q3_k_m.json"
# order in which the L1 patch visits the tensors of one (layer, expert) line
SUFFIX_ORDER = ("gate", "up", "down", "gate_up")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_layers(map_path: Path) -> tuple[dict[int, dict[str, dict[str, int]]], int, dict]:
    doc = json.loads(Path(map_path).read_text(encoding="utf-8"))
    layers: dict[int, dict[str, dict[str, int]]] = {}
    for el in doc["expert_layers"]:
        ent = {}
        for key in SUFFIX_ORDER:
            if key in el and el[key]:
                ent[key] = {"offset": int(el[key]["offset"]), "nb": int(el[key]["nb_expert"]),
                            "n_bytes": int(el[key]["n_bytes"])}
        layers[int(el["layer"])] = ent
    info = {"map": str(map_path), "gguf_basename": doc.get("gguf_basename"), "file_size": int(doc["file_size"]),
            "n_layer": int(doc["n_layer"]), "n_expert": int(doc["n_expert"]),
            "model_sha256": (doc.get("provenance") or {}).get("model", {}).get("sha256")}
    return layers, int(doc["file_size"]), info


def read_hot(path: Path) -> list[tuple[int, int]]:
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 2 and not ln.lstrip().startswith("#"):
            out.append((int(p[0]), int(p[1])))
    return out


class LockModel:
    def __init__(self, layers: dict, file_size: int, n_expert: int, page: int) -> None:
        self.layers, self.size, self.n_expert, self.page = layers, file_size, n_expert, page
        self.n_pages = (file_size + page - 1) // page
        self.locked = bytearray(self.n_pages)      # HotPin's real mlock ranges
        self.needed = bytearray(self.n_pages)      # pages that hold bytes of the listed slices
        self.n_slices = 0
        self.requested_bytes = 0                   # what the tool's "pinned ... MiB" log line counts
        self.rounded_request_bytes = 0
        self.skipped: list[tuple[int, int]] = []
        self.beyond_eof_requests = 0

    def slice_runs(self, layer: int, expert: int) -> list[tuple[int, int, int, int]]:
        """[(lock_first_page, lock_end_page, need_first_page, need_end_page)] for one hot-file line."""
        ent = self.layers.get(layer)
        if ent is None or not (0 <= expert < self.n_expert):
            return []
        P = self.page
        runs = []
        for key in SUFFIX_ORDER:
            t = ent.get(key)
            if not t:
                continue
            off = t["offset"] + expert * t["nb"]
            rounded = (t["nb"] + P - 1) // P * P
            a = off // P
            b = (off + rounded + P - 1) // P
            nb_end = (off + t["nb"] + P - 1) // P
            runs.append((a, min(b, self.n_pages), a, min(nb_end, self.n_pages), t["nb"], rounded, b > self.n_pages))
        return runs  # type: ignore[return-value]

    def new_locked_pages(self, layer: int, expert: int) -> int:
        tmp = 0
        seen: list[tuple[int, int]] = []
        for a, b, *_ in self.slice_runs(layer, expert):
            seg = self.locked[a:b]
            z = seg.count(0)
            # overlap between the runs of the same line (gate_up and down are far apart; keep it exact anyway)
            for sa, sb in seen:
                lo, hi = max(a, sa), min(b, sb)
                if lo < hi:
                    z -= self.locked[lo:hi].count(0)
            seen.append((a, b))
            tmp += z
        return tmp

    def add(self, layer: int, expert: int) -> int:
        runs = self.slice_runs(layer, expert)
        if not runs:
            self.skipped.append((layer, expert))
            return 0
        added = 0
        for a, b, na, nb_, nbytes, rounded, beyond in runs:
            added += self.locked[a:b].count(0)
            self.locked[a:b] = b"\x01" * (b - a)
            self.needed[na:nb_] = b"\x01" * (nb_ - na)
            self.n_slices += 1
            self.requested_bytes += nbytes
            self.rounded_request_bytes += rounded
            self.beyond_eof_requests += int(bool(beyond))
        return added

    def summary(self) -> dict:
        lp, npg = self.locked.count(1), self.needed.count(1)
        return {"page": self.page, "n_slices_locked": self.n_slices,
                "tool_log_requested_bytes": self.requested_bytes,
                "tool_log_requested_mib": round(self.requested_bytes / 2**20, 1),
                "sum_of_rounded_requests_bytes": self.rounded_request_bytes,
                "predicted_unique_locked_pages": lp, "predicted_unique_locked_bytes": lp * self.page,
                "pages_holding_listed_slices": npg, "bytes_holding_listed_slices": npg * self.page,
                "alignment_overlock_pages": lp - npg, "alignment_overlock_bytes": (lp - npg) * self.page,
                "lock_requests_reaching_past_eof": self.beyond_eof_requests,
                "lines_skipped_out_of_range": len(self.skipped)}


def predict(hot: list[tuple[int, int]], map_path: Path, pages: tuple[int, ...] = (4096, 16384)) -> dict:
    layers, size, info = load_layers(map_path)
    out: dict = {"model": info, "n_lines": len(hot), "n_unique_lines": len(set(hot)),
                 "per_layer": [sum(1 for l, _ in hot if l == i) for i in range(info["n_layer"])],
                 "nominal_expert_bytes": 0, "by_page": {}}
    for l, e in set(hot):
        ent = layers.get(l) or {}
        if 0 <= e < info["n_expert"]:
            out["nominal_expert_bytes"] += sum(t["nb"] for t in ent.values())
    for P in pages:
        lm = LockModel(layers, size, info["n_expert"], P)
        for l, e in hot:
            lm.add(l, e)
        out["by_page"][str(P)] = lm.summary()
    return out


def locked_runs(hot: list[tuple[int, int]], map_path: Path, page: int) -> list[tuple[int, int]]:
    """Merged predicted locked page runs [(first_page, end_page)] in file-offset space."""
    layers, size, info = load_layers(map_path)
    lm = LockModel(layers, size, info["n_expert"], page)
    for l, e in hot:
        lm.add(l, e)
    runs, i, n = [], 0, lm.n_pages
    bm = lm.locked
    while True:
        a = bm.find(1, i)
        if a < 0:
            break
        b = bm.find(0, a)
        if b < 0:
            b = n
        runs.append((a, b))
        i = b
    return runs


def run_tool(routing: Path, budget_gb: float, model: Path, out: Path) -> dict:
    argv = [sys.executable, str(HOTPIN_TOOL), "--routing", str(routing), "--budget-gb", repr(float(budget_gb)),
            "--model", str(model), "--out", str(out)]
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    return {"argv": argv, "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
            "tool": str(HOTPIN_TOOL.relative_to(REPO)).replace("\\", "/"), "tool_sha256": sha256_file(HOTPIN_TOOL)}


def write_meta(out: Path, meta: dict) -> None:
    meta["hot_file"] = str(out.relative_to(REPO)).replace("\\", "/") if out.is_absolute() and REPO in out.parents else str(out)
    meta["hot_file_sha256"] = sha256_file(out)
    meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # HotPin's tool prints non-ASCII characters
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("original")
    o.add_argument("--routing", required=True); o.add_argument("--budget-gb", type=float, required=True)
    o.add_argument("--model", required=True); o.add_argument("--out", required=True)
    o.add_argument("--map", default=str(MAP_DEFAULT))
    d = sub.add_parser("order")
    d.add_argument("--routing", required=True); d.add_argument("--model", required=True)
    d.add_argument("--out", required=True); d.add_argument("--map", default=str(MAP_DEFAULT))
    m = sub.add_parser("match")
    m.add_argument("--order", required=True); m.add_argument("--out", required=True)
    m.add_argument("--map", default=str(MAP_DEFAULT)); m.add_argument("--page", type=int, default=4096)
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--expert-pages", type=int); g.add_argument("--total-pages", type=int); g.add_argument("--plan")
    m.add_argument("--mode", choices=["expert", "total"], default="expert",
                   help="with --plan: match C's expert pages (default; HotPin never locks non-expert pages) "
                        "or C's total locked pages")
    p = sub.add_parser("predict")
    p.add_argument("--hot", required=True); p.add_argument("--map", default=str(MAP_DEFAULT))
    a = ap.parse_args()

    if a.cmd in ("original", "order"):
        out = Path(a.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        budget = a.budget_gb if a.cmd == "original" else 1.0e6
        res = run_tool(Path(a.routing), budget, Path(a.model), out)
        sys.stdout.write(res["stdout"])
        if res["rc"] != 0 or not out.exists():
            sys.stderr.write(res["stderr"])
            print("HotPin make_hot_file.py failed", file=sys.stderr)
            return 2
        hot = read_hot(out)
        meta = {"schema": "pigemma-x1b-hotpin-pinset v1",
                "variant": "original (HotPin tool output as is)" if a.cmd == "original"
                           else "full ranking order of HotPin's tool (budget 1e6 GB) - input for 'match'",
                "tool_budget_gb": budget if a.cmd == "original" else None, "tool_run": res,
                "routing": str(a.routing), "routing_sha256": sha256_file(Path(a.routing)),
                "model_arg": str(a.model), "model_bytes": os.path.getsize(a.model),
                "prediction": predict(hot, Path(a.map))}
        write_meta(out, meta)
        print(json.dumps({"lines": len(hot), "by_page": meta["prediction"]["by_page"]}, indent=1))
        return 0

    if a.cmd == "match":
        order = read_hot(Path(a.order))
        target = a.expert_pages if a.expert_pages is not None else a.total_pages
        how = "expert-pages" if a.expert_pages is not None else "total-pages"
        plan_info = None
        if a.plan:
            plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
            if int(plan["page"]) != a.page:
                print(f"plan page {plan['page']} != --page {a.page}", file=sys.stderr)
                return 2
            target = int(plan["expert_pages"]) if a.mode == "expert" else int(plan["total_unique_pages"])
            how = f"plan:{a.mode}"
            plan_info = {"path": str(a.plan), "sha256": sha256_file(Path(a.plan)), "method": plan.get("method"),
                         "n_experts": plan.get("n_experts"), "expert_pages": plan.get("expert_pages"),
                         "nonexpert_pages": plan.get("nonexpert_pages"),
                         "total_unique_pages": plan.get("total_unique_pages")}
        layers, size, info = load_layers(Path(a.map))
        lm = LockModel(layers, size, info["n_expert"], a.page)
        chosen: list[tuple[int, int]] = []
        used = 0
        for l, e in order:
            need = lm.new_locked_pages(l, e)
            if used + need > target:
                break  # strict prefix of HotPin's ranking, like method C
            used += lm.add(l, e)
            chosen.append((l, e))
        out = Path(a.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{l} {e}\n" for l, e in chosen), encoding="ascii", newline="\n")
        meta = {"schema": "pigemma-x1b-hotpin-pinset v1",
                "variant": "LOCALLY ADAPTED budget: strict prefix of HotPin's own ranking order, cut by exact "
                           "predicted unique locked pages instead of the tool's slice estimate",
                "matched_by": how, "target_pages": target, "page": a.page, "plan": plan_info,
                "order_file": str(a.order), "order_sha256": sha256_file(Path(a.order)), "order_lines": len(order),
                "order_exhausted": len(chosen) == len(order),
                "prediction": predict(chosen, Path(a.map))}
        write_meta(out, meta)
        print(json.dumps({"lines": len(chosen), "target_pages": target, "used_pages": used,
                          "order_exhausted": meta["order_exhausted"]}, indent=1))
        return 0

    if a.cmd == "predict":
        print(json.dumps(predict(read_hot(Path(a.hot)), Path(a.map)), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
