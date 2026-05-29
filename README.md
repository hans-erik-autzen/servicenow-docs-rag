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

Clone the repo anywhere you like, then run `install.sh` from that location:

```bash
git clone https://github.com/YOUR_USERNAME/servicenow-docs-rag
cd servicenow-docs-rag
chmod +x install.sh
./install.sh
```

`install.sh` will:

1. Install `uv` if not already present
2. Clone the ServiceNow docs (`australia` release branch) as a **sibling directory** next to this repo
3. Install Python dependencies via `uv sync`
4. Build the ChromaDB vector index — **10–30 minutes on first run**
5. Write the MCP config block to `mcp_config.json` in the repo root

> **Path constraint:** Both repos must be siblings — i.e. in the same parent directory. The folder names
> `servicenow-docs-rag` and `servicenow-docs` must stay as-is. If you need to customise either name,
> an environment variable or config file approach would be required.

## MCP Configuration

After `install.sh` completes, it writes a ready-to-use `mcp_config.json` to the repo root with the correct absolute paths for your machine.

Copy its contents into your Claude Desktop config file:

| OS | Config file path |
|----|-----------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

If you already have other `mcpServers` entries, merge only the `"servicenow-docs"` key into the existing object.

Restart Claude Desktop after saving. The `servicenow-docs` tools will appear automatically.

## Keeping Docs Up to Date

Pull the latest docs and re-index only changed files:

```bash
cd /path/to/servicenow-docs-rag && ./scripts/sync_and_reindex.sh
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
