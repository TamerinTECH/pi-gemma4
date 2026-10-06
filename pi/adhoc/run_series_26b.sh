#!/usr/bin/env bash
# pi-gemma4 ad-hoc: the full 26B-A4B measurement series on the real Pi 5 (run with nohup; survives SSH drops).
#  1. cold, fully logged: 4 short prompts (EN/HE/DE/code) + long EN summarisation prompt
#  2. warm interleaved A/B: no draft vs MTP (--spec-type draft-mtp), A B A B
#  3. memory pressure: mlocked balloon 4 / 6 / 8 GiB (page cache must stream the model from microSD)
set -uo pipefail
cd ~/pigemma
M=models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
MTP=models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf
S=./serve_measure.py
cat > prompts_short2.json <<'EOF'
[{"id": "en_short", "lang": "en", "text": "In three short sentences: why is a mixture-of-experts model faster to run than a dense model of the same size?"},
 {"id": "he_short", "lang": "he", "text": "ענה בעברית בשלושה משפטים קצרים: מה היתרון של מחשב קטן כמו Raspberry Pi?"}]
EOF
echo "=== 1 cold $(date -Is)"
python3 $S --model $M --label 26b-q3km-cold --cold --n-predict 96 --long-file long_en.txt
for r in 1 2; do
  echo "=== 2 A/B round $r $(date -Is)"
  python3 $S --model $M --label 26b-q3km-warm-base-r$r --n-predict 96
  python3 $S --model $M --label 26b-q3km-warm-mtp-r$r --n-predict 96 --extra "-md $MTP --spec-type draft-mtp"
done
for b in 4 6 8; do
  echo "=== 3 balloon ${b} GiB $(date -Is)"
  python3 $S --model $M --label 26b-q3km-balloon${b} --balloon-gib $b --n-predict 48 --prompts prompts_short2.json
done
echo "=== SERIES_DONE $(date -Is)"
df -B1 --output=avail / | tail -1
