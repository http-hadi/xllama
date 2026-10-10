#!/usr/bin/env bash
# ci-publish-logs.sh <branch> <dir> — force-push the (tail of each) text file in <dir> to an orphan branch of this
# repository so the logs can be read without downloading Actions artifacts.
set -euo pipefail
branch="$1"; dir="$2"
[ -d "$dir" ] || { echo "no $dir"; exit 0; }
tmp="$(mktemp -d)"
cp -r "$dir"/. "$tmp"/ 2>/dev/null || true
cd "$tmp"
for f in $(find . -type f -name '*.txt' -o -type f -name '*.csv'); do
  tail -c 60000 "$f" > "$f.tail" && mv "$f.tail" "$f"
done
git init -q . && git checkout -q -b "$branch"
git config user.name ci && git config user.email ci@example.com
git add -A && git commit -q -m "logs ${GITHUB_RUN_ID:-local} ${branch}" || true
git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$branch"
echo "published logs to branch $branch"
