#!/bin/bash

# ============================================================
# pipeline.sh — local dev pipeline for dnd-dm-app
# Usage: ./pipeline.sh "your commit message"
# ============================================================

set -e  # stop on any error

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
  echo ">>> Activating virtual environment..."
  source .venv/bin/activate
fi

COMMIT_MSG=${1:-"auto: update docs and push"}

echo ""
echo "========================================="
echo " DnD pipeline starting..."
echo "========================================="

# ---------------------------------------------------------
# STEP 1 — Run tests
# ---------------------------------------------------------
echo ""
echo ">>> Step 1/3 — Running tests..."
pytest -v

echo ""
echo "✅ All tests passed!"

# ---------------------------------------------------------
# STEP 2 — Update documentation with Claude Code
# ---------------------------------------------------------
echo ""
echo ">>> Step 2/3 — Updating documentation with Claude Code..."

claude --dangerously-skip-permissions "
You are a documentation assistant. Do the following tasks on this project:

1. DOCSTRINGS: Add or update docstrings on all Python functions and classes 
   that are missing them or have outdated ones. Follow Google style docstrings.

2. README.md: Update the README.md to reflect the current state of the project.
   Keep the existing structure but update: feature list, usage instructions, 
   and any outdated information.

3. requirements.txt: Regenerate requirements.txt based on all imports found 
   in the Python files. Only include third-party packages (not stdlib).

4. CHANGELOG.md: Append a new entry at the top of CHANGELOG.md (create the 
   file if it does not exist) with today's date and a summary of recent changes 
   based on recent git commits and current code state.

Keep changes minimal and accurate. Do not invent features that don't exist.
"

echo ""
echo "✅ Documentation updated!"

# ---------------------------------------------------------
# STEP 3 — Commit and push to GitLab
# ---------------------------------------------------------
echo ""
echo ">>> Step 3/3 — Committing and pushing to GitLab..."

git add .
git status

# Only commit if there are changes
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
