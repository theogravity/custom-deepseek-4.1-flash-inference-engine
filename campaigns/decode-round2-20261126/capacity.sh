set -eu
name=$1
python3 qualify.py "$name"
python3 full-record.py "$name"
