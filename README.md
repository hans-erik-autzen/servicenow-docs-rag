# servicenow-docs-rag

Semantic search over the [ServiceNow AI Platform documentation](https://github.com/ServiceNow/ServiceNowDocs) exposed as a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server.

Gives Claude Desktop (or any MCP-compatible client) two tools:

- **`servicenow_docs_search`** — semantic search across all indexed docs
- **`servicenow_docs_get_by_path`** — fetch raw markdown for a specific file

## Prerequisites

- macOS or Linux
- Python 3.10+
- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/) — installed automatically by `install.sh` if missing
- ~3 GB free disk space (docs repo + embeddings)

## Install

The repo must be cloned to `~/servicenow-docs-rag` — the MCP server uses that path.

```bash
git clone https://github.com/YOUR_USERNAME/servicenow-docs-rag ~/servicenow-docs-rag
cd ~/servicenow-docs-rag
chmod +x install.sh
./install.sh
```

`install.sh` will:

1. Install `uv` if not already present
2. Clone the ServiceNow docs (`australia` release branch) to `~/servicenow-docs`
3. Install Python dependencies via `uv sync`
4. Build the ChromaDB vector index — **10–30 minutes on first run**
5. Print the MCP config block to add to Claude Desktop

## MCP Configuration

After `install.sh` completes, add the printed config block to:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "servicenow-docs": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/Users/YOUR_USERNAME/servicenow-docs-rag",
        "python",
        "/Users/YOUR_USERNAME/servicenow-docs-rag/mcp_server/snow_docs_mcp.py"
      ]
    }
  }
}
```

`install.sh` prints this block with the correct paths filled in for your machine — copy it directly from there.

Restart Claude Desktop after saving. The `servicenow-docs` tools will appear automatically.

## Keeping Docs Up to Date

Pull the latest docs and re-index only changed files:

```bash
cd ~/servicenow-docs-rag && ./scripts/sync_and_reindex.sh
```

Unchanged files are skipped via md5 hashing, so subsequent runs are much faster than the initial index.

## Project Layout

```
servicenow-docs-rag/
├── mcp_server/
│   └── snow_docs_mcp.py       # MCP server (FastMCP)
├── scripts/
│   ├── chunk_and_index.py     # Builds/updates the ChromaDB index
│   ├── sync_and_reindex.sh    # Pull latest docs + reindex
│   └── test_mcp_search.py     # Quick search sanity check
├── install.sh                 # One-shot setup script
└── pyproject.toml
```

## How It Works

**Indexing** (`chunk_and_index.py`): splits markdown files on heading boundaries, embeds each chunk with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers, and upserts into a local ChromaDB collection. md5 hashing ensures unchanged files are skipped on re-runs.

**Serving** (`snow_docs_mcp.py`): loads ChromaDB and the embedding model lazily on first query, then serves semantic search and raw file retrieval via [FastMCP](https://github.com/jlowin/fastmcp).
