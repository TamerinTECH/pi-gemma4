"""Summarize independent matched operator runs without hiding per-run variation."""
import argparse
import collections
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('inputs', type=Path, nargs='+')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
groups = collections.defaultdict(list)
checks = []
metadata = []
for path in a.inputs:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    summaries = [r for r in rows if r['record'] == 'summary']
    checks.append(dict(file=str(path), passed=len(summaries) == 1 and summaries[0]['all_same_code_numerical_checks_passed']))
    local = collections.defaultdict(list)
    for row in rows:
        if row['record'] == 'iteration':
            local[row['case'], row['workers'], row['kernel']].append(row)
        if row['record'] in ['metadata', 'case']:
            metadata.append(dict(file=str(path), **row))
    for key, values in local.items():
        def stats(field):
            data = [v[field] for v in values]
            return dict(median=statistics.median(data), mean=statistics.mean(data), minimum=min(data), maximum=max(data))
        groups[key].append(dict(file=str(path), iterations=len(values), full_operator_ms=stats('full_operator_ms'),
                               multiply_dispatch_sync_ms=stats('multiply_dispatch_sync_ms'), reference_q8_prepare_ms=stats('reference_q8_prepare_ms'),
                               numerical_passed=all(v['numerical']['passed'] for v in values)))
result = dict(format='cf-operator-summary-v1', same_code_checks=checks, metadata=metadata,
              timings=[dict(case=k[0], threads=k[1], kernel=k[2], runs=v) for k, v in sorted(groups.items())],
              limits='Logical weight working set larger than L3; no memory-controller traffic counters. Scheduler and output writes included. Native Kleidi quantization is separate in raw records.',
              decision='Timings only; production acceptance and interpretation are recorded in docs/cf-engine/RESULTS.md.')
a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
for key, runs in sorted(groups.items()):
    if key[0].startswith('selected'):
        print(key[1], key[2], ', '.join(f"{r['full_operator_ms']['median']:.4f}" for r in runs))
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
