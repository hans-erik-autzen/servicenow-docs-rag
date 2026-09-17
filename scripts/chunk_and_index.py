#!/usr/bin/env python3
"""
Chunk and index ServiceNow documentation markdown files into ChromaDB.

Splits on heading boundaries, embeds with sentence-transformers, and upserts with
md5-based incremental indexing so unchanged files are skipped on re-runs.

Chunks are sized to fit inside the embedding model's context window. Anything past
that window is silently dropped by sentence-transformers, so `enforce_token_limit`
guarantees the cap holds rather than merely aiming for it.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import chromadb
import frontmatter
import tiktoken
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).parent.parent.resolve()
_DOCS_REPO = _REPO_ROOT.parent / "servicenow-docs"
DOCS_ROOT = _DOCS_REPO / "markdown"
CHROMA_PATH = _REPO_ROOT / "chroma_db"
COLLECTION_NAME = "snow_docs"

# bge-small-en-v1.5 reads 512 wordpiece tokens (vs 256 for all-MiniLM-L6-v2) at the
# same 384 dimensions, so the index stays the same size while nothing gets truncated.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Budget in cl100k tokens. Deliberately well under the model's 512 wordpiece limit:
# wordpiece expands technical text (dotted API names, escaped underscores) past the
# cl100k count, and the title/breadcrumb header eats into the same budget.
MAX_TOKENS = 350

# The cap that actually binds. bge reads 512 wordpieces including [CLS]/[SEP]; anything
# past that is silently dropped at encode time. MAX_TOKENS is only a cheap proxy for this
# — it holds for prose but not for dense technical text, where dotted API names, escaped
# underscores and CSS blocks have been measured expanding 2.2x rather than the ~1.46x the
# proxy assumes. Every chunk is verified against this before it is emitted.
MAX_WORDPIECES = 500

# Below this cl100k count a chunk cannot reach MAX_WORDPIECES, so the slow tokenizer is
# skipped. The lowest cl100k count among measured over-limit chunks was 277.
WORDPIECE_GATE = 250

EMBED_BATCH = 64          # sentence-transformers encode batch size
FLUSH_EVERY = 512         # chunks buffered before an encode + upsert round-trip
CHECKPOINT_EVERY = 200    # files between hash-registry writes
DELETE_BATCH = 200        # paths per collection.delete() call

DEFAULT_RELEASE = "australia"
_release_family = DEFAULT_RELEASE

tokenizer = tiktoken.get_encoding("cl100k_base")
_wordpiece_tokenizer = None

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def set_wordpiece_tokenizer(tok) -> None:
    """Reuse the loaded SentenceTransformer's tokenizer instead of a second copy."""
    global _wordpiece_tokenizer
    _wordpiece_tokenizer = tok


def get_wordpiece_tokenizer():
    """Lazy fallback so chunking is usable (and testable) without building the model."""
    global _wordpiece_tokenizer
    if _wordpiece_tokenizer is None:
        from transformers import AutoTokenizer

        _wordpiece_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _wordpiece_tokenizer


def count_wordpieces(text: str) -> int:
    """Length in the embedding model's own tokens, special tokens included."""
    tok = get_wordpiece_tokenizer()
    # verbose=False suppresses the "longer than maximum sequence length" warning; going
    # over is exactly the condition we are measuring for, not an error.
    return len(tok(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])


def detect_release_family() -> str:
    """Read the checked-out branch of the docs repo instead of hardcoding it."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(_DOCS_REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        branch = proc.stdout.strip()
        if proc.returncode == 0 and branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    return DEFAULT_RELEASE


def pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def release_device_cache(device: str) -> None:
    """Free cached GPU buffers after an encode round-trip.

    The MPS caching allocator keys blocks by tensor shape. Every encode batch
    is padded to a different sequence length, so almost every batch creates a
    new shape and the cache grows across the run instead of reusing memory.
    Left alone it exhausts the MPS budget after ~18k chunks and the process
    dies with "MPS backend out of memory". Releasing the cache after each
    flush keeps it bounded at one flush's worth of buffers.
    """
    if device == "cpu":
        return
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def infer_product_area(path: Path) -> str:
    parts = path.relative_to(DOCS_ROOT).parts
    # parts[-1] is the filename; a file sitting directly under markdown/ has no area.
    return parts[0] if len(parts) > 1 else "unknown"


def build_source_url(path: Path) -> str:
    rel = path.relative_to(_DOCS_REPO)
    return f"https://github.com/ServiceNow/ServiceNowDocs/blob/{_release_family}/{rel}"


def is_toc_file(path: Path, metadata: dict) -> bool:
    """TOC/landing files are unbroken link lists with little semantic value."""
    return path.name == "index.md" or metadata.get("doc_type") == "toc"


def split_by_headings(content: str) -> list[dict]:
    """
    Split markdown into chunks on H1/H2/H3 boundaries.

    Returns dicts of {"text": body, "heading": breadcrumb}. The heading line itself is
    dropped from the body — it is carried in the breadcrumb and re-attached later, so
    keeping it here would duplicate it in the embedded text.
    """
    lines = content.splitlines(keepends=True)

    chunks: list[dict] = []
    current_headings = ["", "", ""]  # h1, h2, h3
    current_lines: list[str] = []
    fence_marker = ""

    def flush(lines_buf: list[str], headings: list[str]) -> None:
        text = "".join(lines_buf).strip()
        if not text:
            return
        breadcrumb = " > ".join(h for h in headings if h)
        chunks.append({"text": text, "heading": breadcrumb})

    for line in lines:
        stripped = line.rstrip("\n")

        fence = _FENCE_RE.match(stripped)
        if fence:
            marker = fence.group(1)
            if not fence_marker:
                fence_marker = marker[0]
            elif marker[0] == fence_marker:
                fence_marker = ""
            current_lines.append(line)
            continue

        # A '#' inside a fenced block is a shell/YAML comment, not a heading.
        m = None if fence_marker else _HEADING_RE.match(stripped)
        if m:
            flush(current_lines, current_headings[:])
            current_lines = []
            level = len(m.group(1))
            current_headings[level - 1] = m.group(2).strip()
            for j in range(level, 3):  # clear deeper levels
                current_headings[j] = ""
        else:
            current_lines.append(line)

    flush(current_lines, current_headings[:])
    return chunks


def _greedy_group(parts: list[str], joiner: str, max_tokens: int) -> list[str]:
    """Greedily pack parts into groups of at most max_tokens (best effort)."""
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        t = count_tokens(part)
        if current_tokens + t > max_tokens and current:
            result.append(joiner.join(current))
            current = [part]
            current_tokens = t
        else:
            current.append(part)
            current_tokens += t

    if current:
        result.append(joiner.join(current))

    return [s for s in result if s.strip()]


def split_on_blank_lines(text: str, max_tokens: int) -> list[str]:
    """
    Split a too-large chunk on blank lines.

    Best effort: a single paragraph with no blank lines is returned whole even if it
    exceeds max_tokens. Use enforce_token_limit when the cap must actually hold.
    """
    return _greedy_group(re.split(r"\n{2,}", text), "\n\n", max_tokens)


def _split_on_token_window(text: str, max_tokens: int) -> list[str]:
    """Last resort: slice on raw token windows so the cap cannot be exceeded."""
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[i:i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _split_within(text: str, max_tokens: int) -> list[str]:
    """
    Split text so every returned part is within max_tokens (cl100k), losing no content.

    Escalates through progressively more aggressive boundaries: blank lines, then
    single newlines, then a hard token window. The final step is what stops an
    unbroken multi-megabyte link list from becoming one giant chunk.
    """
    if count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    parts: list[str] = []
    for para_group in split_on_blank_lines(text, max_tokens):
        if count_tokens(para_group) <= max_tokens:
            parts.append(para_group)
            continue
        for line_group in _greedy_group(para_group.split("\n"), "\n", max_tokens):
            if count_tokens(line_group) <= max_tokens:
                parts.append(line_group)
            else:
                parts.extend(_split_on_token_window(line_group, max_tokens))

    return [p for p in parts if p.strip()]


def _fits_wordpieces(text: str, max_wordpieces: int) -> bool:
    """cl100k first — the real tokenizer only runs where the cheap proxy can't rule it out."""
    if count_tokens(text) <= WORDPIECE_GATE:
        return True
    return count_wordpieces(text) <= max_wordpieces


def enforce_token_limit(
    text: str,
    max_tokens: int,
    max_wordpieces: int = MAX_WORDPIECES,
) -> list[str]:
    """
    Split text so every part fits both the cl100k budget and the model's wordpiece limit.

    The cl100k pass is the fast path and handles all but a fraction of a percent of the
    corpus. Where the proxy under-counts, halve the budget and re-split until the real
    tokenizer agrees — content is redistributed across more parts, never dropped.
    """
    if max_tokens <= 0:
        return [text] if text.strip() else []

    parts = _split_within(text, max_tokens)
    budget = max_tokens
    while budget > 1 and not all(_fits_wordpieces(p, max_wordpieces) for p in parts):
        budget = max(1, budget // 2)
        parts = _split_within(text, budget)

    return parts


def build_header(title: str, breadcrumb: str) -> str:
    """Context line prepended to a chunk's embedded text."""
    if title and breadcrumb:
        # The H1 usually repeats the frontmatter title; don't say it twice.
        if breadcrumb == title or breadcrumb.startswith(f"{title} > "):
            return breadcrumb
        return f"{title} > {breadcrumb}"
    return title or breadcrumb


def compose_chunks(
    raw_chunks: list[dict],
    title: str,
    description: str,
    max_tokens: int = MAX_TOKENS,
    max_wordpieces: int = MAX_WORDPIECES,
) -> list[dict]:
    """
    Attach title/breadcrumb context to each chunk and enforce the token cap.

    The header goes on every part, not just the first, so continuation chunks stay
    retrievable on their own.
    """
    composed: list[dict] = []

    for chunk in raw_chunks:
        header = build_header(title, chunk["heading"])
        body = chunk["text"]

        # The description only adds signal where there is no heading context at all.
        if description and not chunk["heading"]:
            body = f"{description}\n\n{body}"

        prefix = f"{header}\n\n" if header else ""
        budget = max_tokens - count_tokens(prefix)
        # The header is prepended after the split, so its wordpiece cost has to come out
        # of the budget too — otherwise a chunk that fit on its own goes over once
        # prefixed. Counting special tokens on both sides just leaves a little headroom.
        wp_budget = max_wordpieces - count_wordpieces(prefix) if prefix else max_wordpieces
        if budget <= 0 or wp_budget <= 0:  # pathological header; drop it for this chunk
            prefix, budget, wp_budget = "", max_tokens, max_wordpieces

        for part in enforce_token_limit(body, budget, wp_budget):
            composed.append({"text": f"{prefix}{part}".strip(), "heading": chunk["heading"]})

    return composed


def make_chunk_id(path: Path, index: int) -> str:
    rel = str(path.relative_to(_DOCS_REPO))
    return hashlib.md5(f"{rel}::{index}".encode()).hexdigest()


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


_FM_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def parse_frontmatter_loosely(text: str) -> tuple[dict, str]:
    """
    Recover simple scalar keys from a YAML block that strict parsing rejected.

    ~1,000 upstream files carry an unquoted title containing a colon
    ('title: Example: WS-Security SOAP envelope header'), which is invalid YAML.
    Giving up on those would drop the title — the strongest retrieval signal we
    have — so scan the block line by line instead.

    Returns (metadata, body). Both are empty/unchanged when there is no block.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text

    block, body = parts[1], parts[2].lstrip()
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line[0] in " \t-":  # nested mapping or list item
            continue
        m = _FM_KEY_RE.match(line)
        if not m:
            continue
        value = m.group(2).strip().strip("\"'")
        if value and value[0] not in "[{":  # scalars only
            meta[m.group(1)] = value

    return meta, body


def delete_paths(collection, paths: list[str]) -> int:
    """Remove every chunk belonging to the given docs-relative paths."""
    removed = 0
    for i in range(0, len(paths), DELETE_BATCH):
        batch = paths[i:i + DELETE_BATCH]
        collection.delete(where={"path": {"$in": batch}})
        removed += len(batch)
    return removed


def main() -> None:
    global _release_family
    _release_family = detect_release_family()

    if not DOCS_ROOT.is_dir():
        raise SystemExit(
            f"Docs not found at {DOCS_ROOT}\n"
            "Run ./install.sh, or clone ServiceNow/ServiceNowDocs next to this repo."
        )

    device = pick_device()
    print(f"Loading embedding model ({EMBED_MODEL}) on {device}…", flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=device)
    # Chunking verifies against the model's own vocabulary, not a second downloaded copy.
    set_wordpiece_tokenizer(model.tokenizer)

    print(f"Connecting to ChromaDB at {CHROMA_PATH}", flush=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        # bge is a cosine model; Chroma defaults to l2, which ranks differently.
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )

    existing_model = (collection.metadata or {}).get("embed_model")
    if existing_model and existing_model != EMBED_MODEL:
        raise SystemExit(
            f"Existing index was built with '{existing_model}', but this script uses "
            f"'{EMBED_MODEL}'.\nVector spaces are not comparable — delete {CHROMA_PATH} "
            "and rebuild."
        )

    hash_registry_path = CHROMA_PATH / "file_hashes.json"
    if hash_registry_path.exists():
        hash_registry: dict[str, str] = json.loads(hash_registry_path.read_text())
    else:
        hash_registry = {}

    md_files = sorted(DOCS_ROOT.rglob("*.md"))
    total_files = len(md_files)
    print(f"Found {total_files} markdown files under {DOCS_ROOT}", flush=True)

    files_processed = 0
    files_skipped = 0
    toc_skipped = 0
    chunks_created = 0
    errors: list[str] = []
    indexable_paths: set[str] = set()

    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_metas: list[dict] = []

    def flush_pending() -> int:
        nonlocal pending_ids, pending_texts, pending_metas
        if not pending_ids:
            return 0
        embeddings = model.encode(
            pending_texts,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        release_device_cache(device)
        collection.upsert(
            ids=pending_ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=pending_texts,
            metadatas=pending_metas,
        )
        n = len(pending_ids)
        pending_ids, pending_texts, pending_metas = [], [], []
        return n

    for file_path in md_files:
        rel_str = str(file_path.relative_to(_DOCS_REPO))

        # Cheap check first — avoids parsing frontmatter for every TOC file.
        if file_path.name == "index.md":
            toc_skipped += 1
            continue

        current_hash = file_md5(file_path)
        seen_before = rel_str in hash_registry

        if hash_registry.get(rel_str) == current_hash:
            indexable_paths.add(rel_str)
            files_skipped += 1
            continue

        try:
            post = frontmatter.load(str(file_path))
            content = post.content
            fm = post.metadata or {}
        except Exception as e:
            reason = str(e).splitlines()[0]
            fm, content = parse_frontmatter_loosely(file_path.read_text(errors="replace"))
            errors.append(
                f"{rel_str}: invalid YAML frontmatter ({reason}); "
                f"recovered {sorted(fm)[:4]}"
            )

        if is_toc_file(file_path, fm):
            toc_skipped += 1
            if seen_before:  # previously indexed, now excluded
                delete_paths(collection, [rel_str])
                hash_registry.pop(rel_str, None)
            continue

        indexable_paths.add(rel_str)

        title = str(fm.get("title") or "").strip()
        description = str(fm.get("description") or "").strip()
        last_updated = str(fm.get("last_updated") or "").strip()

        final_chunks = compose_chunks(split_by_headings(content), title, description)

        # Chunk ids are md5(path::index), so a file that now yields fewer chunks would
        # leave the surplus behind. Clear the path before re-adding it. The pending
        # buffer never holds this path yet (its chunks are appended below), so the
        # delete cannot race with a buffered write.
        if seen_before:
            delete_paths(collection, [rel_str])

        product_area = infer_product_area(file_path)
        source_url = build_source_url(file_path)

        for idx, chunk in enumerate(final_chunks):
            text = chunk["text"].strip()
            if not text:
                continue
            pending_ids.append(make_chunk_id(file_path, idx))
            pending_texts.append(text)
            pending_metas.append({
                "path": rel_str,
                "title": title,
                "release_family": _release_family,
                "section_heading": chunk["heading"],
                "product_area": product_area,
                "source_url": source_url,
                "last_updated": last_updated,
                "char_count": len(text),
            })

        if len(pending_ids) >= FLUSH_EVERY:
            chunks_created += flush_pending()

        hash_registry[rel_str] = current_hash
        files_processed += 1

        if files_processed % CHECKPOINT_EVERY == 0:
            chunks_created += flush_pending()
            # Checkpoint so an interrupted run doesn't re-embed everything next time.
            hash_registry_path.write_text(json.dumps(hash_registry, indent=2))
            print(
                f"  {files_processed} files indexed, {chunks_created} chunks so far…",
                flush=True,
            )

    chunks_created += flush_pending()

    # Purge chunks for files that were deleted upstream or are no longer indexable.
    stale = sorted(set(hash_registry) - indexable_paths)
    if stale:
        print(f"Removing {len(stale)} stale files from the index…", flush=True)
        delete_paths(collection, stale)
        for rel in stale:
            hash_registry.pop(rel, None)

    hash_registry_path.write_text(json.dumps(hash_registry, indent=2))

    print("\n=== Indexing complete ===")
    print(f"  Files indexed   : {files_processed}")
    print(f"  Files unchanged : {files_skipped}")
    print(f"  TOC files skipped: {toc_skipped}")
    print(f"  Stale removed   : {len(stale)}")
    print(f"  Chunks created  : {chunks_created}")
    print(f"  Errors          : {len(errors)}")
    for e in errors[:20]:
        print(f"    {e}")
    if len(errors) > 20:
        print(f"    … and {len(errors) - 20} more")
    print(f"  ChromaDB total  : {collection.count()} chunks")


if __name__ == "__main__":
    main()
