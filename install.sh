#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$(dirname "$REPO_DIR")/servicenow-docs"
DOCS_REPO="https://github.com/ServiceNow/ServiceNowDocs"
DOCS_BRANCH="australia"

# Install uv if missing
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Clone or update ServiceNow docs
if [[ -d "$DOCS_DIR/.git" ]]; then
  echo "Updating ServiceNow docs..."
  git -C "$DOCS_DIR" pull
else
  echo "Cloning ServiceNow docs ($DOCS_BRANCH branch)..."
  git clone --depth 1 --branch "$DOCS_BRANCH" "$DOCS_REPO" "$DOCS_DIR"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
uv sync

# Build the vector index
echo ""
echo "Building ChromaDB index..."
echo "(First run takes 10–30 minutes depending on your machine)"
echo ""
uv run python scripts/chunk_and_index.py

# Write MCP config to file
MCP_CONFIG_FILE="$REPO_DIR/mcp_config.json"
cat > "$MCP_CONFIG_FILE" <<EOF
{
  "mcpServers": {
    "servicenow-docs": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "$REPO_DIR",
        "python",
        "$REPO_DIR/mcp_server/snow_docs_mcp.py"
      ]
    }
  }
}
EOF

cat <<EOF

===================================================
 Setup complete!
===================================================

MCP config written to:
  $MCP_CONFIG_FILE

Copy its contents into:
  ~/Library/Application Support/Claude/claude_desktop_config.json

If you already have other mcpServers entries, merge only the
"servicenow-docs" key into the existing mcpServers object.

Then restart Claude Desktop.
EOF
