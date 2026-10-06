import json
from pathlib import Path

out = Path('results/cf-engine/diagnostic-inputs')
out.mkdir(parents=True, exist_ok=True)
modes = ['reference', 'reference', 'dotprod', 'asm', 'reference']
rows = [dict(id=f'{i}-{m}', prompt='Explain in one sentence why ice floats on water.', max_tokens=4) for i, m in enumerate(modes)]
(out/'prompts.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8', newline='\n')
(out/'pairs.json').write_text(json.dumps([dict(id='neon', reference=1, candidate=2), dict(id='asm', reference=1, candidate=3)], indent=2)+'\n', encoding='utf-8')
