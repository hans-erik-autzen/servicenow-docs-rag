#!/usr/bin/env python3
"""
Chunk and index ServiceNow documentation markdown files into ChromaDB.
Splits on heading boundaries, embeds with sentence-transformers, upserts with md5-based
incremental indexing so unchanged files are skipped on re-runs.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import chromadb
import frontmatter
import tiktoken
from sentence_transformers import SentenceTransformer

DOCS_ROOT = Path.home() / "servicenow-docs" / "markdown"
CHROMA_PATH = Path.home() / "servicenow-docs-rag" / "chroma_db"
COLLECTION_NAME = "snow_docs"
GITHUB_BASE = "https://github.com/ServiceNow/ServiceNowDocs/blob/australia"
RELEASE_FAMILY = "australia"
MAX_TOKENS = 1200

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def infer_product_area(path: Path) -> str:
    parts = path.relative_to(DOCS_ROOT).parts
    return parts[0] if parts else "unknown"


def build_source_url(path: Path) -> str:
    rel = path.relative_to(Path.home() / "servicenow-docs")
    return f"{GITHUB_BASE}/{rel}"


def split_by_headings(content: str) -> list[dict]:
    """Split markdown content into chunks on H1/H2/H3 boundaries."""
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    lines = content.splitlines(keepends=True)

    chunks = []
    current_headings = ["", "", ""]  # h1, h2, h3
    current_lines: list[str] = []
    current_start = 0

    def flush(lines_buf: list[str], headings: list[str]) -> None:
        text = "".join(lines_buf).strip()
        if not text:
            return
        breadcrumb = " > ".join(h for h in headings if h)
        if breadcrumb:
            full_text = f"{breadcrumb}\n\n{text}"
        else:
            full_text = text
        chunks.append({"text": full_text, "heading": breadcrumb})

    for i, line in enumerate(lines):
        m = heading_re.match(line.rstrip("\n"))
        if m:
            flush(current_lines, current_headings[:])
            current_lines = [line]
            level = len(m.group(1))
            title = m.group(2).strip()
            current_headings[level - 1] = title
            # clear deeper levels
            for j in range(level, 3):
                current_headings[j] = ""
        else:
            current_lines.append(line)

    flush(current_lines, current_headings[:])
    return chunks


def split_on_blank_lines(text: str, max_tokens: int) -> list[str]:
    """Further split a too-large chunk on blank lines."""
    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        t = count_tokens(para)
        if current_tokens + t > max_tokens and current_parts:
            result.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = t
        else:
            current_parts.append(para)
            current_tokens += t

    if current_parts:
        result.append("\n\n".join(current_parts))

    return [s for s in result if s.strip()]


def make_chunk_id(path: Path, index: int) -> str:
    rel = str(path.relative_to(Path.home() / "servicenow-docs"))
    return hashlib.md5(f"{rel}::{index}".encode()).hexdigest()


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    print("Loading embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Load existing hash registry from a sidecar JSON to avoid querying chroma per-file
    hash_registry_path = CHROMA_PATH / "file_hashes.json"
    if hash_registry_path.exists():
        hash_registry: dict[str, str] = json.loads(hash_registry_path.read_text())
    else:
        hash_registry = {}

    md_files = sorted(DOCS_ROOT.rglob("*.md"))
    total_files = len(md_files)
    print(f"Found {total_files} markdown files under {DOCS_ROOT}")

    files_processed = 0
    files_skipped = 0
    chunks_created = 0
    errors: list[str] = []

    for file_path in md_files:
        rel_str = str(file_path.relative_to(Path.home() / "servicenow-docs"))
        current_hash = file_md5(file_path)

        if hash_registry.get(rel_str) == current_hash:
            files_skipped += 1
            continue

        try:
            post = frontmatter.load(str(file_path))
            content = post.content
        except Exception:
            content = file_path.read_text(errors="replace")

        product_area = infer_product_area(file_path)
        source_url = build_source_url(file_path)

        raw_chunks = split_by_headings(content)

        final_chunks: list[dict] = []
        for chunk in raw_chunks:
            if count_tokens(chunk["text"]) > MAX_TOKENS:
                sub_parts = split_on_blank_lines(chunk["text"], MAX_TOKENS)
                for sub in sub_parts:
                    final_chunks.append({"text": sub, "heading": chunk["heading"]})
            else:
                final_chunks.append(chunk)

        ids, embeddings, documents, metadatas = [], [], [], []
        for idx, chunk in enumerate(final_chunks):
            text = chunk["text"].strip()
            if not text:
                continue
            chunk_id = make_chunk_id(file_path, idx)
            embedding = model.encode(text).tolist()
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append({
                "path": rel_str,
                "release_family": RELEASE_FAMILY,
                "section_heading": chunk["heading"],
                "product_area": product_area,
                "source_url": source_url,
                "char_count": len(text),
            })

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            chunks_created += len(ids)

        hash_registry[rel_str] = current_hash
        files_processed += 1

        if files_processed % 50 == 0:
            print(f"  {files_processed}/{total_files - files_skipped} files processed, "
                  f"{chunks_created} chunks so far…")

    # Persist hash registry
    hash_registry_path.write_text(json.dumps(hash_registry, indent=2))

    print("\n=== Indexing complete ===")
    print(f"  Files processed : {files_processed}")
    print(f"  Files skipped   : {files_skipped}")
    print(f"  Chunks created  : {chunks_created}")
    print(f"  Errors          : {len(errors)}")
    if errors:
        for e in errors:
            print(f"    {e}")
    print(f"  ChromaDB total  : {collection.count()} chunks")


if __name__ == "__main__":
    main()
