set -eu
p=/tmp/ds4-c1-1126
name=$1
cd "$p/$name/work"
export CARGO_TARGET_DIR=/tmp/ds4-cpu-eval/target
cargo fmt --check
cargo build --release --locked --offline > ../build.log 2>&1
python3 /tmp/ds4-cpu-eval/image.py c1-$name > ../image.log 2>&1
cargo test --release --locked --offline --message-format=json > ../tests.json 2> ../tests.log
cargo clippy --release --locked --offline -- -D warnings > ../clippy.log 2>&1
