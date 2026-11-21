#!/usr/bin/env bash
# Prepare a release up to a draft GitHub release:
#   scripts/release.sh X.Y.Z --notes NOTES.md
# Pins runtime-host.json to the runtime release of the same version on a
# branch (release the runtime first), lands it through a pull request once CI
# passes, promotes the merged develop commit to main (dry run first) and
# creates a draft release for vX.Y.Z on that commit. Publishing the draft tags
# the release and starts the release workflows; that stays a manual step.
# Needs gh with rights to merge pull requests and run workflows, and a Python
# with packaging installed (PYTHON, default python3).
set -euo pipefail

usage() { echo "usage: $0 X.Y.Z --notes FILE" >&2; exit 2; }
[ $# -eq 3 ] && [ "$2" = --notes ] || usage
VERSION="$1"
NOTES="$(realpath "$3")"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
[ -s "$NOTES" ] || { echo "empty or missing notes: $NOTES" >&2; exit 2; }

REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
BRANCH="release/$VERSION"
ROOT="$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python3}"
# A relative path has to keep working after the cd into the worktree.
[[ "$PYTHON" != */* || "$PYTHON" = /* ]] || PYTHON="$PWD/$PYTHON"

# The release commit: the bundled host runtime pinned to runtime vVERSION. The
# compiler version itself comes from the tag, so nothing else changes.
prepare() {
    "$PYTHON" scripts/runtime_bundle.py --pin "v$VERSION"
}
COMMIT_SUBJECT="release: pin the v$VERSION host runtime"

# Waits until every CI check run on the commit has completed, then requires each
# to have passed. Runs register over the first minute, so it waits that out.
wait_for_ci() {
    local sha="$1" runs deadline=$((SECONDS + 3600))
    echo "Waiting for CI on $sha"
    sleep 60
    while :; do
        # Leaves out promote.yml's own check run, which lands on develop's head.
        runs="$(gh api --paginate "repos/$REPO/commits/$sha/check-runs" \
            --jq '.check_runs[] | select(.name != "promote") | [.status, (.conclusion // "")] | @tsv')"
        if [ -n "$runs" ] && ! grep -qv '^completed' <<<"$runs"; then
            break
        fi
        [ "$SECONDS" -lt "$deadline" ] || { echo "CI on $sha did not complete within 60 min" >&2; exit 1; }
        sleep 30
    done
    if grep -qvE $'^completed\t(success|skipped|neutral)$' <<<"$runs"; then
        echo "CI failed on $sha" >&2
        exit 1
    fi
}

promote() {
    local sha="$1" dry_run="$2" last run_id deadline=$((SECONDS + 300))
    # The dispatched run is the first one with a larger id; the local clock plays no part.
    last="$(gh run list -R "$REPO" -w promote.yml -L 1 --json databaseId --jq '.[0].databaseId // 0')"
    gh workflow run promote.yml -R "$REPO" --ref develop \
        -f sha="$sha" -f version="$VERSION" -f dry_run="$dry_run"
    run_id=""
    while [ -z "$run_id" ]; do
        [ "$SECONDS" -lt "$deadline" ] || { echo "the dispatched promote run did not appear" >&2; exit 1; }
        sleep 5
        run_id="$(gh run list -R "$REPO" -w promote.yml -L 5 --json databaseId \
            --jq "[.[] | select(.databaseId > $last)] | min_by(.databaseId) | .databaseId // empty")"
    done
    timeout 3600 gh run watch "$run_id" -R "$REPO" --exit-status >/dev/null \
        || { echo "promote (dry_run=$dry_run) failed: run $run_id" >&2; exit 1; }
    echo "promote (dry_run=$dry_run) passed: run $run_id"
}

git -C "$ROOT" fetch --quiet origin
WORKTREE="$(mktemp -d)"
git -C "$ROOT" worktree add --quiet -b "$BRANCH" "$WORKTREE" origin/develop
# The branch can only be deleted once no worktree has it checked out.
trap 'git -C "$ROOT" worktree remove --force "$WORKTREE" && git -C "$ROOT" branch --quiet -D "$BRANCH"' EXIT
cd "$WORKTREE"
prepare
if git diff --quiet; then
    echo "develop already carries $VERSION; promoting its head"
    SHA="$(git rev-parse origin/develop)"
else
    git commit --quiet -am "$COMMIT_SUBJECT"
    # A rerun reuses the branch an earlier run pushed when it holds the same
    # change on the same base; any other remote branch makes the push fail.
    pushed="$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
    if [ -n "$pushed" ] && [ "$(git rev-parse "$pushed^{tree}" "$pushed^")" \
        = "$(git rev-parse "HEAD^{tree}" HEAD^)" ]; then
        git reset --quiet --hard "$pushed"
    else
        git push --quiet -u origin "$BRANCH"
    fi
    head="$(git rev-parse HEAD)"
    # -H matches by branch name only, so a fork's pull request can share it.
    PR="$(gh pr list -R "$REPO" -B develop -H "$BRANCH" --json number,headRefOid,isCrossRepository \
        --jq "[.[] | select(.headRefOid == \"$head\" and (.isCrossRepository | not))][0].number // empty")"
    [ -n "$PR" ] || PR="$(gh pr create -R "$REPO" -B develop -H "$BRANCH" -t "$COMMIT_SUBJECT" \
        -b "Prepares the $VERSION release." | grep -oE '[0-9]+$')"
    wait_for_ci "$head"
    gh pr merge "$PR" -R "$REPO" --rebase --match-head-commit "$head"
    SHA="$(gh pr view "$PR" -R "$REPO" --json mergeCommit --jq .mergeCommit.oid)"
fi
wait_for_ci "$SHA"
promote "$SHA" true
promote "$SHA" false
[ "$(gh api "repos/$REPO/commits/main" --jq .sha)" = "$SHA" ] \
    || { echo "main did not move to $SHA" >&2; exit 1; }
gh release create "v$VERSION" -R "$REPO" --draft --target "$SHA" \
    --title "v$VERSION" --notes-file "$NOTES"
echo "Draft release v$VERSION is ready on $SHA. Publishing it tags the release."
