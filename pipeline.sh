#!/bin/bash

# ============================================================
# pipeline.sh — lean pipeline
# Documentation enforced at coding time via CLAUDE.md
# Usage: ./pipeline.sh "your commit message"
# ============================================================

set -e

COMMIT_MSG=${1:-"auto: commit and push"}

echo ""
echo "========================================="
echo " Pipeline starting..."
echo "========================================="

# ---------------------------------------------------------
# Activate venv if present
# ---------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
  echo ">>> Activating virtual environment..."
  source .venv/bin/activate
fi

# ---------------------------------------------------------
# STEP 1 — Run tests
# ---------------------------------------------------------
echo ""
echo ">>> Step 1/3 — Running tests..."
pytest -v

echo ""
echo "✅ All tests passed!"

# ---------------------------------------------------------
# STEP 2 — Check requirements and structure
# ---------------------------------------------------------
echo ""
echo ">>> Step 2/3 — Checking requirements and structure..."

# Check if any Python files changed
CHANGED_PY=$(git diff --name-only --diff-filter=d HEAD -- '*.py' | tr '\n' ' ')

if [ -n "$CHANGED_PY" ]; then
  echo ">>> Changed Python files: $CHANGED_PY"

  # Update requirements.txt from venv if present, skip otherwise
  if [ -f ".venv/bin/activate" ]; then
    echo ">>> Updating requirements.txt from venv..."
    pip freeze | grep -v '^\-e' > requirements.txt
  else
    echo "ℹ️  No venv found — skipping requirements.txt update."
  fi

  # Check if project structure changed (files added or deleted)
  STRUCTURE_CHANGES=$(git diff --name-only --diff-filter=AD HEAD)
  if [ -n "$STRUCTURE_CHANGES" ]; then
    echo ">>> Structure changes detected: $STRUCTURE_CHANGES"
    claude --dangerously-skip-permissions "
Files were added or deleted in this project: $STRUCTURE_CHANGES

TASKS (be minimal and token-efficient):
1. Update README.md only if the structure change affects usage or features.
2. Append ONE brief entry to CHANGELOG.md with today's date summarizing
   the structural change. Create the file if it does not exist.
Do not touch any other files.
"
  else
    echo "ℹ️  No structure changes — skipping README/CHANGELOG update."
  fi

else
  echo "ℹ️  No Python files changed — skipping requirements check."
fi

echo ""
echo "✅ Requirements and structure check done!"

# ---------------------------------------------------------
# STEP 3 — Commit and push
# ---------------------------------------------------------
echo ""
echo ">>> Step 3/3 — Committing and pushing..."

git add .
git status

if git diff --staged --quiet; then
  echo "ℹ️  No changes to commit."
else
  git commit -m "$COMMIT_MSG"
  git push gitlab
  echo ""
  echo "✅ Pushed to GitLab!"
fi

echo ""
echo "========================================="
echo " Pipeline complete! 🎉"
echo "========================================="
