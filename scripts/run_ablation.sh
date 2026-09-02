#!/bin/zsh
# Experiment controlat: aceleasi date, aceiasi hiperparametri, aceeasi samanta.
# Singura diferenta: ramura de dezlegare holografica (circcorr).
cd "$(dirname "$0")/.."
while pgrep -f "scripts/train.py" > /dev/null; do sleep 5; done

# linia de baza tocmai s-a terminat: o marcam explicit ca fiind fara dezlegare
python3 - <<'PY'
import json
p = "checkpoints/rezonet.cfg.json"
c = json.load(open(p)); c["unbind"] = False
json.dump(c, open("checkpoints/base.cfg.json", "w"))
PY
for e in npz vocab.json history.json; do cp "checkpoints/rezonet.$e" "checkpoints/base.$e"; done
echo "=== LINIA DE BAZA salvata in checkpoints/base.* (fara dezlegare) ==="

python3 scripts/train.py --steps 2500 --batch 32 --block 128 --d_model 128 \
  --layers 3 --d_bind 160 --lr 4e-3 --eval_every 125 \
  --out checkpoints/rezonet > checkpoints/train_unbind.log 2>&1
echo "=== VARIANTA CU DEZLEGARE antrenata ==="
