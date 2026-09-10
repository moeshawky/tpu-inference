#!/usr/bin/env bash
# Read-only repository state census helper

set -euo pipefail

echo "========================================"
echo "      REPOSITORY STATE CENSUS"
echo "========================================"

CANONICAL_BRANCH="i-j-port"
echo "Policy Canonical Serving Branch: ${CANONICAL_BRANCH}"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "Not a git repository")
if [ "$REPO_ROOT" = "Not a git repository" ]; then
    echo "Error: Not inside a git repository."
else
    echo "Repository Root: ${REPO_ROOT}"

    ORIGIN_URL=$(git remote get-url origin 2>/dev/null || echo "none")
    echo "Origin URL: ${ORIGIN_URL}"

    UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "none")
    echo "Upstream URL: ${UPSTREAM_URL}"

    HEAD_SHA=$(git rev-parse HEAD)
    echo "Current HEAD Full SHA: ${HEAD_SHA}"

    CURRENT_BRANCH=$(git branch --show-current)
    if [ -z "$CURRENT_BRANCH" ]; then
        CURRENT_BRANCH="(detached HEAD)"
    fi
    echo "Current Branch/State: ${CURRENT_BRANCH}"

    DIRTY_STATUS=$(git status --porcelain)
    if [ -z "$DIRTY_STATUS" ]; then
        echo "Working Tree: clean"
    else
        echo "Working Tree: dirty"
    fi

    echo "---"
    echo "Local Branches:"
    git branch --format="%(refname:short)" || true

    echo "---"
    echo "Remote Branches:"
    git branch -r --format="%(refname:short)" || true

    echo "---"
    echo "Upstream Tracking & Ahead/Behind:"
    git for-each-ref --format="%(refname:short) %(upstream:short) %(upstream:track)" refs/heads || true

    echo "---"
    echo "Worktrees:"
    git worktree list || true

    echo "---"
    echo "Canonical Branch Checks:"

    if [ "$CURRENT_BRANCH" = "$CANONICAL_BRANCH" ]; then
        echo "Current HEAD is on the canonical serving branch (${CANONICAL_BRANCH})."
    else
        echo "Current HEAD is NOT on the canonical serving branch."
    fi

    CANONICAL_TIP=$(git rev-parse "refs/heads/${CANONICAL_BRANCH}" 2>/dev/null || git rev-parse "refs/remotes/origin/${CANONICAL_BRANCH}" 2>/dev/null || echo "")

    if [ -z "$CANONICAL_TIP" ]; then
        echo "Canonical branch tip not found locally."
    elif [ "$HEAD_SHA" = "$CANONICAL_TIP" ]; then
        echo "Current HEAD SHA EQUALS the canonical branch tip."
    else
        echo "Current HEAD SHA DOES NOT equal the canonical branch tip."
    fi

    if [ "$CURRENT_BRANCH" != "$CANONICAL_BRANCH" ]; then
        echo "========================================"
        echo "WARNING: EXPERIMENTAL / NON-CANONICAL CHECKOUT"
        echo "The current checkout is NOT the canonical serving branch."
        echo "Success while running here does not indicate canonical readiness."
        echo "========================================"
    fi
fi
