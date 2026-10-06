"""Sync the reviewed CF status into CLAUDE while retaining historical constraints."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
target = root / 'CLAUDE.md'
current = target.read_text(encoding='utf-8')
marker = '## Historical decision brief (2026-10-05 evening)'
if marker not in current:
    marker = '## CURRENT BRIEF (2026-10-05 evening)'
history = current[current.index(marker):].replace(marker, '## Historical decision brief (2026-10-05 evening)', 1)
head = (root / 'docs/cf-engine/STATUS.md').read_text(encoding='utf-8')
assert not any(ord(c) < 32 and c not in '\n\t\r' for c in head)
target.write_text(head + '\n' + history, encoding='utf-8', newline='\n')
