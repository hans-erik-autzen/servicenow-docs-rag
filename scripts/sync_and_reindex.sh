#!/bin/bash
set -euo pipefail

cd ~/servicenow-docs && git pull
cd ~/servicenow-docs-rag && uv run python scripts/chunk_and_index.py
