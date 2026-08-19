#!/usr/bin/env python3
"""
MCP server exposing ServiceNow documentation search via ChromaDB.

Tools:
  - servicenow_docs_search            : semantic search over indexed chunks
  - servicenow_docs_list_product_areas: valid product_area filter values
  - servicenow_docs_get_by_path       : return raw markdown for a repo path
"""

from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).parent.parent.resolve()
CHROMA_PATH = _REPO_ROOT / "chroma_db"
DOCS_ROOT = (_REPO_ROOT.parent / "servicenow-docs").resolve()
COLLECTION_NAME = "snow_docs"

# bge models expect this instruction on queries only — never on indexed passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_RESULTS_CAP = 25
MAX_RESULT_CHARS = 4000
MAX_FILE_CHARS = 100_000

REBUILD_HINT = "Run: uv run python scripts/chunk_and_index.py"

mcp = FastMCP("servicenow-docs")

# Lazy-loaded singletons
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_embed_model_name: str | None = None
_distance_space: str = "cosine"


def _get_collection() -> chromadb.Collection:
    global _collection, _embed_model_name, _distance_space

    if _collection is not None:
        return _collection

    if not CHROMA_PATH.is_dir():
        raise RuntimeError(f"No index found at {CHROMA_PATH}. {REBUILD_HINT}")

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not open the '{COLLECTION_NAME}' collection: {e}. {REBUILD_HINT}"
        ) from e

    metadata = collection.metadata or {}
    model_name = metadata.get("embed_model")
    if not model_name:
        # Querying with a different model than the index was built with silently
        # produces nonsense, so refuse rather than guess.
        raise RuntimeError(
            "Index does not record which embedding model built it, so it predates the "
            f"current indexer and cannot be queried safely. {REBUILD_HINT}"
        )

    _collection = collection
    _embed_model_name = model_name
    _distance_space = metadata.get("hnsw:space", "l2")
    return _collection


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _get_collection()  # resolves the model name recorded at index time
        _model = SentenceTransformer(_embed_model_name)
    return _model


def _to_score(distance: float) -> float | None:
    """Convert a raw distance to a 0-1 similarity, when the metric allows it."""
    if distance is None:
        return None
    if _distance_space == "cosine":
        return round(1.0 - distance, 4)
    if _distance_space == "ip":
        return round(-distance, 4)
    return None  # l2 distances have no meaningful 0-1 mapping


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n… [truncated, {len(text) - limit} more characters]"


@mcp.tool()
def servicenow_docs_search(
    query: str,
    product_area: str | None = None,
    max_results: int = 6,
) -> list[dict]:
    """
    Search ServiceNow AI Platform documentation semantically.

    Args:
        query: Natural language search query.
        product_area: Optional filter. Must be an exact value from
            servicenow_docs_list_product_areas — an unrecognised value silently
            matches nothing.
        max_results: Number of results to return (default 6, max 25).

    Returns:
        List of dicts with keys: text, score, source_url, section_heading, title,
        product_area, path. Results are ordered best-first; score is 0-1 similarity.
    """
    model = _get_model()
    collection = _get_collection()

    n_results = max(1, min(int(max_results), MAX_RESULTS_CAP))

    text_to_embed = query
    if "bge" in (_embed_model_name or "").lower():
        text_to_embed = f"{BGE_QUERY_PREFIX}{query}"

    embedding = model.encode(text_to_embed, normalize_embeddings=True).tolist()
    where = {"product_area": product_area} if product_area else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    output = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        distance = distances[i] if i < len(distances) else None
        output.append({
            "text": _truncate(doc, MAX_RESULT_CHARS),
            "score": _to_score(distance),
            "title": meta.get("title", ""),
            "source_url": meta.get("source_url", ""),
            "section_heading": meta.get("section_heading", ""),
            "product_area": meta.get("product_area", ""),
            "path": meta.get("path", ""),
        })

    return output


@mcp.tool()
def servicenow_docs_list_product_areas() -> list[str]:
    """
    List the valid values for the product_area filter of servicenow_docs_search.

    Call this before filtering — passing an unrecognised area returns no results
    rather than an error.
    """
    markdown_root = DOCS_ROOT / "markdown"
    if not markdown_root.is_dir():
        raise RuntimeError(
            f"Docs not found at {markdown_root}. Run ./install.sh, or clone "
            "ServiceNow/ServiceNowDocs next to this repo."
        )
    return sorted(p.name for p in markdown_root.iterdir() if p.is_dir())


@mcp.tool()
def servicenow_docs_get_by_path(path: str) -> str:
    """
    Return the raw markdown content of a ServiceNow documentation file.

    Args:
        path: Relative path from the docs repo root, as returned in a search
              result's `path` field, e.g. 'markdown/now-platform/index.md'

    Returns:
        Raw markdown text of the file, or an error message if not found.
    """
    # Resolve before checking: is_relative_to is a lexical test, so an unresolved
    # path containing '..' would pass it and still escape DOCS_ROOT.
    full_path = (DOCS_ROOT / path).resolve()

    if not full_path.is_relative_to(DOCS_ROOT):
        return "Path traversal not allowed."
    if not full_path.is_file():
        return f"File not found: {path}"

    return _truncate(full_path.read_text(errors="replace"), MAX_FILE_CHARS)


if __name__ == "__main__":
    mcp.run(transport="stdio")
