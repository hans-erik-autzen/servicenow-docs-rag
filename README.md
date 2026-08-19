# servicenow-docs-rag

Semantic search over the [ServiceNow AI Platform documentation](https://github.com/ServiceNow/ServiceNowDocs) exposed as a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server.

Indexes ~50,000 documentation files into ~307,000 locally-embedded chunks and gives Claude Desktop (or any MCP-compatible client) three tools.

## Tools

### `servicenow_docs_search`

Semantic search across all indexed docs.

| Parameter | Default | Notes |
|---|---|---|
| `query` | — | Natural language query |
| `product_area` | none | Optional filter. Must be an exact value from `servicenow_docs_list_product_areas` — an unrecognised value silently matches nothing |
| `max_results` | 6 | Clamped to 25 |

Returns results best-first, each with `text`, `score` (0–1 cosine similarity), `title`, `section_heading`, `product_area`, `path`, and `source_url` (a GitHub permalink). Result text is capped at 4,000 characters.

### `servicenow_docs_list_product_areas`

The 56 valid values for the `product_area` filter, read from the docs tree. Call this before filtering.

### `servicenow_docs_get_by_path`

Fetches raw markdown for one file, given a `path` as returned in a search result (e.g. `markdown/now-platform/index.md`). Reads are confined to the docs repo; output is capped at 100,000 characters.

## Prerequisites

- macOS or Linux
- Python 3.10+
- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/) — installed automatically by `install.sh` if missing
- **~5 GB free disk space**

Measured after a full install:

| | Size |
|---|---|
| Docs repo (`../servicenow-docs`) | 508 MB |
| ChromaDB index (`chroma_db/`) | 2.8 GB |
| Python environment (`.venv/`) | 930 MB |
| Embedding model (cached in `~/.cache/huggingface`) | 128 MB |

The embedding model downloads automatically on first use — no manual step, but the first run needs network access.

## Install

Clone the repo anywhere you like, then run `install.sh` from that location:

```bash
git clone https://github.com/hans-erik-autzen/servicenow-docs-rag
cd servicenow-docs-rag
chmod +x install.sh
./install.sh
```

`install.sh` will:

1. Install `uv` if not already present
2. Clone the ServiceNow docs (`australia` release branch) as a **sibling directory** next to this repo
3. Install Python dependencies via `uv sync`
4. Build the ChromaDB vector index — **expect roughly 40 minutes** (measured on an Apple M5 Pro with MPS acceleration; CPU-only machines will be slower)
5. Write the MCP config block to `mcp_config.json` in the repo root

The indexer prints progress every 200 files and checkpoints as it goes, so an interrupted run resumes rather than starting over.

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

Restart Claude Desktop after saving. The three `servicenow-docs` tools will appear automatically.

## Verifying the Install

Smoke-test search without a running MCP server:

```bash
uv run python scripts/test_mcp_search.py
```

Run the unit tests (chunking logic — no index required):

```bash
uv run pytest tests/ -q
```

## Keeping Docs Up to Date

Pull the latest docs and re-index only changed files:

```bash
cd /path/to/servicenow-docs-rag && ./scripts/sync_and_reindex.sh
```

Unchanged files are skipped via md5 hashing, so subsequent runs are far faster than the initial index. Re-runs also **remove** chunks for files that shrank or were deleted upstream, so the index does not accumulate results pointing at content that no longer exists.

## Project Layout

```
servicenow-docs-rag/
├── mcp_server/
│   └── snow_docs_mcp.py       # MCP server (FastMCP)
├── scripts/
│   ├── chunk_and_index.py     # Builds/updates the ChromaDB index
│   ├── sync_and_reindex.sh    # Pull latest docs + reindex
│   └── test_mcp_search.py     # Quick search sanity check
├── tests/
│   └── test_chunking.py       # Unit tests for chunking + frontmatter parsing
├── install.sh                 # One-shot setup script
└── pyproject.toml
```

## How It Works

**Indexing** (`chunk_and_index.py`): splits markdown files on heading boundaries, embeds each chunk with [`bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via sentence-transformers, and upserts into a local ChromaDB collection using cosine distance. md5 hashing ensures unchanged files are skipped on re-runs.

Details that matter for retrieval quality:

- **Chunks are sized to the model.** `bge-small-en-v1.5` reads 512 wordpiece tokens and silently
  discards anything beyond that, so chunks are capped at 350 tokens (`MAX_TOKENS`) and the cap is
  enforced by splitting — on blank lines, then newlines, then a hard token window — rather than
  merely aimed for.
- **Every chunk carries its context.** Each one is prefixed with `title > heading breadcrumb`,
  including continuation chunks, so it can be retrieved on its own.
- **`index.md` / `doc_type: toc` files are skipped.** They are unbroken lists of links with little
  semantic value (the largest was 2.2 MB). They remain reachable via `servicenow_docs_get_by_path`.
- **Malformed frontmatter is recovered, not discarded.** ~1,030 upstream files have invalid YAML
  (an unquoted `title:` containing a colon). Their scalar keys are parsed line-by-line rather than
  dropped, so those files keep their title.
- **Headings inside fenced code blocks are ignored**, so a `#` comment in a shell sample does not
  split a chunk.

**Serving** (`snow_docs_mcp.py`): loads ChromaDB and the embedding model lazily on first query,
then serves search and raw file retrieval via [FastMCP](https://github.com/jlowin/fastmcp) over
stdio. The server reads the embedding model name recorded in the collection's metadata and loads
that exact model, so the index and the server cannot silently drift into incompatible vector spaces.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: ../servicenow-docs/` | Sibling docs repo not cloned | Run `./install.sh` or manually clone `ServiceNow/ServiceNowDocs` next to this repo |
| Search returns 0 results | Index not built yet | Run `uv run python scripts/chunk_and_index.py` |
| Indexing reports `Errors : 1029` | Upstream files with invalid YAML frontmatter | **Expected, not a failure.** Those files are indexed via a fallback parser that recovers their metadata |
| MCP server not visible in Claude Desktop | Config not copied or path is wrong | Re-run `./install.sh`, then copy the printed JSON block into `claude_desktop_config.json` and restart Claude Desktop |
| `ModuleNotFoundError` on any import | Dependencies not installed | Run `uv sync` |
| First query is very slow | Embedding model downloads on first use | Expected — model is cached after that; subsequent queries are fast |
| `Index does not record which embedding model built it` | Index predates the current indexer | `rm -rf chroma_db` and run `uv run python scripts/chunk_and_index.py` |
| `Existing index was built with '<model>'` | `EMBED_MODEL` changed since the index was built | `rm -rf chroma_db` and rebuild — vector spaces are not comparable |
| Filtering by `product_area` returns nothing | Value is not a real product area | Call `servicenow_docs_list_product_areas` for the exact set |
