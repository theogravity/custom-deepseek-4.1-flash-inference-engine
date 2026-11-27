set -eu
p=/tmp/ds4-tool-return-1126
cd "$p/router"
# Build after the preceding timed jobs return the GPUs to the stable service.
exec 9>>/tmp/ds4-native-bench.lock
flock -x 9
test -f "$p/qualification/passed.json"
export CARGO_TARGET_DIR=/tmp/ds4-cpu-eval/target
cd "$p/idle"
cargo build --release --locked --offline > "$p/idle-rebuild.log" 2>&1
python3 "$p/provenance.py"
cd "$p/router"
cargo fmt --check
cargo test --release --locked --offline > "$p/router-tests.log" 2>&1
cargo clippy --all-targets --release --locked --offline -- -D warnings > "$p/router-clippy.log" 2>&1
cargo build --release --locked --offline > "$p/router-build.log" 2>&1
python3 /tmp/ds4-cpu-eval/image.py pi-bound-router-1126 > "$p/router-image.log" 2>&1
flock -u 9
python3 "$p/canonical.py" router bound-off-a bound-on bound-off-b
python3 "$p/post_canonical.py" bound-off-a bound-on bound-off-b
