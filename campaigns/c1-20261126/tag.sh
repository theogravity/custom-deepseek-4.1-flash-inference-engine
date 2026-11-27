set -eu
p=/tmp/ds4-c1-1126
name=$1
tag=$2
decision=$3
message=$4
cd "$p/$name/work"
git add src native "results/c1-$name.json"
git diff --cached --check
git commit -m "$message"
git tag -a "$tag" -m "$message"
cd /home/benchmark/llm/ds4-flash-rs
if [ "$decision" = keep ]; then
 git merge --ff-only "$tag"
else
 git restore --source "$tag" -- "results/c1-$name.json"
 git add "results/c1-$name.json"
 git commit -m "Record $name single-client experiment results"
fi
