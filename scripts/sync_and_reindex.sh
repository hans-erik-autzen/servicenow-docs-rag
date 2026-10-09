#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$(dirname "$REPO_DIR")/servicenow-docs"

git -C "$DOCS_DIR" pull
# Extra args (e.g. --branch australia) pass through to the indexer.
uv run --project "$REPO_DIR" python "$REPO_DIR/scripts/chunk_and_index.py" "$@"
