"""Check the preserved five-request numerical diagnostic, including reference repeats."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_sweep import BinaryRecords

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
result = []
with BinaryRecords(a.directory/'logits.bin', 'logits') as logits, BinaryRecords(a.directory/'routes.bin', 'routes') as routes:
    for left, right, name in [(0, 1, 'reference-repeat-before'), (1, 4, 'reference-repeat-after'), (1, 2, 'reference-neon'), (1, 3, 'reference-asm')]:
        lrows, rrows = logits.records[left], logits.records[right]
        same_keys = [(r['token'], r['vocab']) for r in lrows] == [(r['token'], r['vocab']) for r in rrows]
        mismatch = 0
        maximum = 0.0
        for l, r in zip(lrows, rrows):
            lv = np.frombuffer(logits.mapping, dtype='<f4', count=l['vocab'], offset=l['offset']).copy()
            rv = np.frombuffer(logits.mapping, dtype='<f4', count=r['vocab'], offset=r['offset']).copy()
            mismatch += int(np.count_nonzero(lv.view('<u4') != rv.view('<u4')))
            maximum = max(maximum, float(np.max(np.abs(lv.astype('f8')-rv))))
        def route_values(request):
            return [(r['name'], r['slots'], r['tokens'], bytes(routes.mapping[r['offset']:r['offset']+r['slots']*r['tokens']*4])) for r in routes.records[request]]
        result.append(dict(comparison=name, logit_keys_match=same_keys, logit_bit_mismatches=mismatch,
                           logit_max_absolute=maximum, routing_records_exact=route_values(left)==route_values(right)))
a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
