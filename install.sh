#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/servicenow-docs-rag"
DOCS_DIR="$HOME/servicenow-docs"
DOCS_REPO="https://github.com/ServiceNow/ServiceNowDocs"
DOCS_BRANCH="australia"

# Verify the repo was cloned to the expected location
if [[ "$(pwd)" != "$REPO_DIR" ]]; then
  echo "Error: This script must be run from $REPO_DIR"
  echo ""
  echo "Clone the repo to the correct location first:"
  echo "  git clone https://github.com/YOUR_USERNAME/servicenow-docs-rag $REPO_DIR"
  echo "  cd $REPO_DIR && ./install.sh"
  exit 1
fi

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

# Print MCP config
cat <<EOF

===================================================
 Setup complete! Add this to your Claude Desktop
 config and restart Claude Desktop.
===================================================

Config file: ~/Library/Application Support/Claude/claude_desktop_config.json

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

If you already have other mcpServers entries, add only the
"servicenow-docs" key inside the existing mcpServers object.
EOF
