#!/usr/bin/env python3
"""
MCP server exposing ServiceNow documentation search via ChromaDB.
Tools:
  - servicenow_docs_search   : semantic search over indexed chunks
  - servicenow_docs_get_by_path : return raw markdown for a repo path
"""

import os
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path.home() / "servicenow-docs-rag" / "chroma_db"
DOCS_ROOT = Path.home() / "servicenow-docs"
COLLECTION_NAME = "snow_docs"

mcp = FastMCP("servicenow-docs")

# Lazy-loaded singletons
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


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
        product_area: Optional filter (e.g. 'now-assist', 'ai-search', 'intelligent-experiences').
        max_results: Number of results to return (default 6).

    Returns:
        List of dicts with keys: text, source_url, section_heading, product_area, path.
    """
    model = _get_model()
    collection = _get_collection()

    embedding = model.encode(query).tolist()
    where = {"product_area": product_area} if product_area else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=max_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        output.append({
            "text": doc,
            "source_url": meta.get("source_url", ""),
            "section_heading": meta.get("section_heading", ""),
            "product_area": meta.get("product_area", ""),
            "path": meta.get("path", ""),
        })

    return output


@mcp.tool()
def servicenow_docs_get_by_path(path: str) -> str:
    """
    Return the raw markdown content of a ServiceNow documentation file.

    Args:
        path: Relative path from the repo root, e.g.
              'markdown/now-assist/index.md'

    Returns:
        Raw markdown text of the file, or an error message if not found.
    """
    full_path = DOCS_ROOT / path
    if not full_path.exists():
        return f"File not found: {path}"
    if not full_path.is_relative_to(DOCS_ROOT):
        return "Path traversal not allowed."
    return full_path.read_text(errors="replace")


if __name__ == "__main__":
    mcp.run(transport="stdio")
