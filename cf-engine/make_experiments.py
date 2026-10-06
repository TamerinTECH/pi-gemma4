"""Generate public/synthetic request lists; no model execution or private data."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--kernel', choices=['asm', 'asm-unroll2', 'asm-prefetch512'], default='asm')
p.add_argument('--include-neon', action='store_true', help='Compare both NEON and assembly to each held-out reference request')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
prompt = 'Reply in one sentence: what does a thermometer measure?'
order = [a.kernel, 'reference', 'dotprod', a.kernel, a.kernel, 'dotprod', 'reference', 'dotprod', 'reference', a.kernel]
perf = [{'id': f'{i:02d}-{mode}', 'prompt': prompt, 'max_tokens': 32} for i, mode in enumerate(order)]
quality = [
    ('en-short', 'Explain in one sentence why ice floats on water.', 48),
    ('en-extract', 'Record: Ada bought 7 blue pens and 3 red pens. Return only the number of blue pens.', 12),
    ('en-negation', 'Anna did not buy tea. She bought coffee. What did Anna buy? Return only the item.', 12),
    ('de-number', 'Berechne 17 + 25. Antworte nur mit der Zahl.', 12),
    ('de-negation', 'Lena fährt nicht nach Berlin, sondern nach Bonn. Wohin fährt Lena? Nur der Ortsname.', 12),
    ('he-number', 'כמה הם 17 ועוד 25? החזר רק מספר.', 12),
    ('he-extract', 'דנה קנתה שלושה תפוחים ושתי אגסים. כמה תפוחים קנתה דנה? השב רק במספר.', 12),
    ('code', 'Write only a one-line Python function named square that returns the square of x. No markdown.', 32),
    ('long-extract', 'Read the following synthetic inventory. '+ ' '.join(f'Box {i} contains {i+3} pencils.' for i in range(20))+' How many pencils are in Box 13? Return only the number.', 12),
    ('topic-change', 'We were discussing metal temperature. New task: What is the capital of Germany? Return only the city name.', 12),
]
qrows = [dict(id='warmup', prompt=prompt, max_tokens=32)]
qorder = [a.kernel]
pairing = []
for i, (name, text, limit) in enumerate(quality):
    modes = ['reference', 'dotprod', a.kernel] if a.include_neon else ['reference', a.kernel]
    if i % 2:
        modes.reverse()
    pair = {}
    for mode in modes:
        pair[mode] = len(qrows)
        qrows.append(dict(id=f'{name}-{mode}', prompt=text, max_tokens=limit))
        qorder.append(mode)
    for mode in modes:
        if mode != 'reference':
            pairing.append(dict(id=f'{name}-{mode}', reference=pair['reference'], candidate=pair[mode]))
for name, rows, schedule in [('perf', perf, order), ('quality', qrows, qorder)]:
    (a.out/f'{name}-prompts.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (a.out/f'{name}-schedule.txt').write_text(','.join(schedule)+'\n', encoding='ascii', newline='\n')
(a.out/'quality-pairs.json').write_text(json.dumps(pairing, indent=2)+'\n', encoding='utf-8')
