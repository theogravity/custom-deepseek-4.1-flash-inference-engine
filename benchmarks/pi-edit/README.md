# Pi coding task

Pi reads a failing Go flag-parser test, edits the parser and reruns the tests. The grader checks the final tree in a fresh sandbox with an additional private-to-the-agent test. The prompts and grader in this directory were used for the published agentic coding comparison. The fixture intentionally starts with a failing test. [Task, controls and results](../../AGENTIC-CODING.md)

The included provider file and extension reproduce the workstation's Pi configuration. The driver requires Pi release 1.4.0, a sandbox-compatible Node.js binary (tested with the official Linux x64 distribution of v22.23.3), Go, bubblewrap and socat on Linux. The engine serves on port 8001. Set `LITELLM_MASTER_KEY` in the calling shell. `PI_RELEASE_DIR` and `PI_NODE_BINARY` override the Pi release directory and Node executable. Run from a private temporary copy of this directory. The driver records the first failing-test request there for replay.

```sh
umask 077
run_dir=$(mktemp -d)
cp -R benchmarks/pi-edit/. "$run_dir"
cd "$run_dir"
mkdir cache
# Set PI_NODE_BINARY to the extracted bin/node if the host Node needs private libraries.
python3 setup.py
python3 -c 'import json,os,trial; print(json.dumps(trial.one(os.environ["LITELLM_MASTER_KEY"], 0), indent=2))'
```
