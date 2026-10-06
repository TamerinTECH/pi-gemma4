"""Independently compare captured logits/routing bytes; never execute a model.

python cf-engine/check_quality_bits.py --directory results/cf-engine/exact-quality \
  --pairs results/cf-engine/exact-suite/quality-pairs.json \
  --out results/cf-engine/exact-quality-bits.json

Uses only the standard library, no numerical conversion/comparison and no imports
from the main analyzer. Payload reads are bounded to 64 KiB; indexes grow with
record count. Request IDs are paired explicitly, not compared as payload bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys

CHUNK = 65536


def natural(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def index_dump(path, kind):
    """Sequential independent parse, checksum and finite/nonnegative validation."""
    records, seen = defaultdict(list), set()
    digest = hashlib.sha256()
    size = path.stat().st_size
    if not size:
        raise ValueError(f"{path}: empty dump")
    with path.open("rb") as stream:
        def take(count):
            if not 0 <= count <= CHUNK:
                raise ValueError("internal unbounded read")
            data = stream.read(count)
            if len(data) != count:
                raise ValueError(f"{path}: truncated dump at {stream.tell()}")
            digest.update(data)
            return data

        while stream.tell() < size:
            if kind == "logits":
                vocab, request, token = struct.unpack("<QQQ", take(24))
                if not 1 <= vocab <= 1048576 or (request, token) in seen:
                    raise ValueError(f"{path}: invalid vocabulary or duplicate request/token")
                seen.add((request, token))
                row = {"token": token, "vocab": vocab}
                count = vocab
            else:
                name_length, slots, tokens, request = struct.unpack("<QQQQ", take(32))
                if not (1 <= name_length <= 4096 and 1 <= slots <= 128 and 1 <= tokens <= 1048576):
                    raise ValueError(f"{path}: invalid route dimensions")
                count = slots*tokens
                if count > 16777216:
                    raise ValueError(f"{path}: excessive route payload")
                row = {"name": take(name_length).decode("utf-8"), "slots": slots, "tokens": tokens}
            row.update(offset=stream.tell(), bytes=count*4)
            if count*4 > size-stream.tell():
                raise ValueError(f"{path}: truncated {kind} payload")
            remaining = count*4
            while remaining:
                data = take(min(remaining, CHUNK))
                if kind == "logits":
                    # Inspect IEEE-754 exponent bits: catch NaN/Inf even when
                    # both captures contain the same nonfinite bit pattern.
                    invalid = any(word & 0x7f800000 == 0x7f800000 for (word,) in struct.iter_unpack("<I", data))
                else:
                    invalid = any(word < 0 for (word,) in struct.iter_unpack("<i", data))
                if invalid:
                    raise ValueError(f"{path}: nonfinite logit or negative expert ID in request {request}")
                remaining -= len(data)
            records[request].append(row)
    return records, {"path": path.as_posix(), "bytes": size, "sha256": digest.hexdigest(),
                     "records": sum(map(len, records.values())), "requests": len(records), "exact_end_reached": True}


def compare_payloads(left_file, right_file, left, right):
    left_file.seek(left["offset"])
    right_file.seek(right["offset"])
    offset, first = 0, None
    mismatching_bytes = 0
    while offset < left["bytes"]:
        count = min(CHUNK, left["bytes"]-offset)
        a, b = left_file.read(count), right_file.read(count)
        if len(a) != count or len(b) != count:
            raise ValueError("dump changed/truncated during comparison")
        if a != b:
            for index, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    mismatching_bytes += 1
                    if first is None:
                        first = {"payload_byte": offset+index, "word": (offset+index)//4,
                                 "reference_byte": x, "candidate_byte": y}
        offset += count
    return mismatching_bytes, first


def compare_records(path, index, reference, candidate, keys):
    left, right = index.get(reference, []), index.get(candidate, [])
    compared, words, byte_mismatches, first = 0, 0, 0, None
    with path.open("rb") as a_file, path.open("rb") as b_file:
        for ordinal, (a, b) in enumerate(itertools.zip_longest(left, right)):
            if a is None or b is None or any(a[key] != b[key] for key in keys):
                if first is None:
                    first = {"record": ordinal, "reason": "record count/order/shape/name mismatch"}
                continue
            unequal, position = compare_payloads(a_file, b_file, a, b)
            compared += 1
            words += a["bytes"]//4
            byte_mismatches += unequal
            if position is not None and first is None:
                first = {"record": ordinal, "reason": "raw payload bytes differ", **position}
    return {"passed": bool(left) and bool(right) and first is None,
            "reference_records": len(left), "candidate_records": len(right),
            "paired_records_compared": compared, "words_compared": words,
            "mismatching_payload_bytes": byte_mismatches, "first_mismatch": first}


def check(directory, pairs_path):
    pairs = load_json(pairs_path)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs must be a nonempty list")
    requests = {}
    with (directory/"requests.jsonl").open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            request = natural(row["chunk_id"], "request")
            if request in requests:
                raise ValueError("duplicate request record")
            requests[request] = natural(row["n_sampled"], "sample count", 1)
    logits, logit_identity = index_dump(directory/"logits.bin", "logits")
    routes, route_identity = index_dump(directory/"routes.bin", "routes")
    if set(logits) != set(requests) or set(routes) != set(requests):
        raise ValueError("request coverage differs between dumps and completed requests")
    for request, expected in requests.items():
        if [record["token"] for record in logits[request]] != list(range(expected)):
            raise ValueError(f"request {request}: missing/extra/out-of-order sampled-token logits")
    profile = load_json(directory/"profile.json")
    expected_counts = Counter()
    for operation in profile.get("ops", []):
        if operation.get("op") == "MUL_MAT_ID" and operation.get("weight_name"):
            expected_counts[operation["weight_name"]] += natural(operation["calls"], "profile calls")
    actual_counts = Counter(row["name"] for group in routes.values() for row in group)
    if not expected_counts or actual_counts != expected_counts:
        raise ValueError("route projection counts differ from profile; capture may omit whole records")
    checked, seen_ids, seen_pairs = [], set(), set()
    for pair in pairs:
        identifier = pair["id"]
        a, b = natural(pair["reference"], "reference"), natural(pair["candidate"], "candidate")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids or (a, b) in seen_pairs:
            raise ValueError("empty/duplicate pair ID or request pair")
        if a == b or a not in requests or b not in requests:
            raise ValueError("pair requires two distinct complete requests")
        seen_ids.add(identifier)
        seen_pairs.add((a, b))
        logits_result = compare_records(directory/"logits.bin", logits, a, b, ("token", "vocab", "bytes"))
        routes_result = compare_records(directory/"routes.bin", routes, a, b, ("name", "slots", "tokens", "bytes"))
        checked.append({"id": identifier, "reference": a, "candidate": b,
                        "passed": logits_result["passed"] and routes_result["passed"],
                        "logits": logits_result, "routes": routes_result})
    return {"passed": all(pair["passed"] for pair in checked),
            "directory": directory.as_posix(), "pairs_path": pairs_path.as_posix(),
            "requested_pairs": len(pairs), "checked_pairs": len(checked),
            "files": {"logits": logit_identity, "routes": route_identity},
            "all_sampled_tokens_covered": True, "route_projection_counts_match_profile": True,
            "totals": {kind: {field: sum(pair[kind][field] for pair in checked)
                for field in ("paired_records_compared", "words_compared", "mismatching_payload_bytes")}
                for kind in ("logits", "routes")}, "pairs": checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = {"format": "cf-engine-quality-bits", "version": 1, "passed": False, "errors": [],
              "maximum_payload_read_bytes": CHUNK,
              "comparison": "exact raw float32 and ordered int32 bytes, including signed-zero bits",
              "scope": "captured paired payloads only; no universal bit identity, task quality, or sampling-distribution claim; use analyze_sweep.py for input/backend/output validation"}
    status = 2
    try:
        result.update(check(args.directory, args.pairs))
        status = 0 if result["passed"] else 1
    except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
        result["errors"].append(str(error))
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    if args.out:
        protected = [args.directory/name for name in ("logits.bin", "routes.bin", "requests.jsonl", "profile.json")]
        if args.out.resolve() in {path.resolve() for path in protected+[args.pairs]}:
            parser.error("output must not overwrite an input")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
        print(json.dumps({"passed": result["passed"], "out": args.out.as_posix(),
                          "totals": result.get("totals"), "errors": result["errors"]}))
    else:
        print(encoded, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
