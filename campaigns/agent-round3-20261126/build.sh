set -eu
cd "/tmp/ds4-agent-round3/$1"
export CARGO_TARGET_DIR=/tmp/ds4-cpu-eval/target
cargo fmt
cargo test --release --locked --offline > "../$1-tests.log" 2>&1
cargo clippy --all-targets --release --locked --offline -- -D warnings > "../$1-clippy.log" 2>&1
cargo build --release --locked --offline > "../$1-build.log" 2>&1
python3 /tmp/ds4-cpu-eval/image.py "agent3-$1" > "../$1-image.log" 2>&1
