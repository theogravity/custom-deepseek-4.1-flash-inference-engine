set -eu
name=$1
python3 check-extra.py "$name"
python3 extra.py "$name" repeat edit long
python3 run.py "$name" repeat-config.json
python3 extra.py "$name" repeat
python3 heldout.py "$name"
