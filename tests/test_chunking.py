import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from chunk_and_index import (
    MAX_TOKENS,
    _REPO_ROOT,
    build_header,
    compose_chunks,
    count_tokens,
    enforce_token_limit,
    infer_product_area,
    make_chunk_id,
    split_by_headings,
    split_on_blank_lines,
    parse_frontmatter_loosely,
)

_DOCS_ROOT = _REPO_ROOT.parent / "servicenow-docs"
_FAKE_PATH = _DOCS_ROOT / "markdown" / "test_file.md"


class TestSplitByHeadings:
    def test_no_headings_returns_single_chunk(self):
        chunks = split_by_headings("Just some text\nwith no headings.")
        assert len(chunks) == 1
        assert "Just some text" in chunks[0]["text"]
        assert chunks[0]["heading"] == ""

    def test_single_h1(self):
        chunks = split_by_headings("# Title\n\nSome content here.")
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Title"
        assert "Some content here" in chunks[0]["text"]

    def test_h1_and_h2_breadcrumb(self):
        chunks = split_by_headings("# H1\n\nIntro.\n\n## H2\n\nSection content.")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "H1"
        assert chunks[1]["heading"] == "H1 > H2"

    def test_h1_h2_h3_nesting(self):
        chunks = split_by_headings("# H1\n\nTop.\n\n## H2\n\nMid.\n\n### H3\n\nDeep.")
        assert len(chunks) == 3
        assert chunks[0]["heading"] == "H1"
        assert chunks[1]["heading"] == "H1 > H2"
        assert chunks[2]["heading"] == "H1 > H2 > H3"

    def test_consecutive_headings_no_empty_chunks(self):
        chunks = split_by_headings("# H1\n## H2\n## H3\n\nContent.")
        for chunk in chunks:
            assert chunk["text"].strip() != ""

    def test_content_before_first_heading(self):
        chunks = split_by_headings("Preamble text.\n\n# Title\n\nBody.")
        assert chunks[0]["heading"] == ""
        assert "Preamble text" in chunks[0]["text"]

    def test_heading_line_not_duplicated_in_body(self):
        # The heading is carried in the breadcrumb and re-attached by compose_chunks;
        # leaving it in the body too would embed it twice.
        chunks = split_by_headings("# Title\n\nBody text.")
        assert chunks[0]["text"] == "Body text."

    def test_deeper_level_reset_on_new_h2(self):
        chunks = split_by_headings("# H1\n\n## A\n\n### Deep.\n\nx\n\n## B\n\ny")
        assert chunks[-1]["heading"] == "H1 > B"

    def test_hash_inside_fenced_code_is_not_a_heading(self):
        content = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\nAfter."
        chunks = split_by_headings(content)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Real"
        assert "# not a heading" in chunks[0]["text"]

    def test_tilde_fence_also_respected(self):
        content = "# Real\n\n~~~yaml\n# comment\n~~~\n\nAfter."
        chunks = split_by_headings(content)
        assert len(chunks) == 1
        assert "# comment" in chunks[0]["text"]

    def test_heading_after_fence_closes_still_splits(self):
        content = "# One\n\n```\n# fake\n```\n\n## Two\n\nbody"
        chunks = split_by_headings(content)
        assert [c["heading"] for c in chunks] == ["One", "One > Two"]


class TestSplitOnBlankLines:
    def test_short_text_returned_as_single_chunk(self):
        text = "Short paragraph."
        result = split_on_blank_lines(text, max_tokens=500)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_split_into_multiple_chunks(self):
        para = "word " * 10  # ~10 tokens each
        text = f"{para}\n\n{para}\n\n{para}"
        result = split_on_blank_lines(text, max_tokens=15)
        assert len(result) > 1
        for chunk in result:
            assert chunk.strip() != ""

    def test_single_oversized_paragraph_not_dropped(self):
        # split_on_blank_lines is best-effort: with no blank line to split on it
        # returns the paragraph whole. enforce_token_limit is what binds the cap.
        text = "word " * 500
        result = split_on_blank_lines(text, max_tokens=10)
        assert len(result) == 1
        assert "word" in result[0]


class TestEnforceTokenLimit:
    def test_short_text_untouched(self):
        assert enforce_token_limit("Short.", 100) == ["Short."]

    def test_every_part_within_cap(self):
        para = "word " * 40
        text = "\n\n".join([para] * 5)
        for part in enforce_token_limit(text, 25):
            assert count_tokens(part) <= 25

    def test_unbroken_paragraph_is_split(self):
        # The 2.2 MB index.md chunk was a single unbroken line; this is the case
        # that previously escaped the cap entirely.
        text = "word " * 2000
        parts = enforce_token_limit(text, 50)
        assert len(parts) > 1
        for part in parts:
            assert count_tokens(part) <= 50

    def test_no_content_lost_when_splitting(self):
        text = "alpha beta gamma delta " * 200
        parts = enforce_token_limit(text, 30)
        rejoined = "".join(parts)
        assert rejoined.count("alpha") == text.count("alpha")

    def test_single_long_line_without_spaces(self):
        text = "x" * 20000
        parts = enforce_token_limit(text, 40)
        for part in parts:
            assert count_tokens(part) <= 40
        assert "".join(parts) == text

    def test_newline_only_text_is_split(self):
        text = "\n".join(f"- link {i}" for i in range(500))
        for part in enforce_token_limit(text, 60):
            assert count_tokens(part) <= 60


class TestBuildHeader:
    def test_title_and_breadcrumb_joined(self):
        assert build_header("Doc", "H1 > H2") == "Doc > H1 > H2"

    def test_title_not_repeated_when_breadcrumb_starts_with_it(self):
        assert build_header("Doc", "Doc > H2") == "Doc > H2"

    def test_title_equal_to_breadcrumb(self):
        assert build_header("Doc", "Doc") == "Doc"

    def test_missing_pieces(self):
        assert build_header("Doc", "") == "Doc"
        assert build_header("", "H1") == "H1"
        assert build_header("", "") == ""


class TestComposeChunks:
    def test_header_prepended(self):
        raw = [{"text": "Body.", "heading": "H1 > H2"}]
        out = compose_chunks(raw, "Doc", "")
        assert out[0]["text"].startswith("Doc > H1 > H2")
        assert "Body." in out[0]["text"]

    def test_every_subchunk_keeps_the_header(self):
        # Continuation chunks must stand alone; without the header they lose all
        # topical context and become unretrievable.
        body = "\n\n".join(["word " * 60] * 8)
        raw = [{"text": body, "heading": "H1 > H2"}]
        out = compose_chunks(raw, "Doc", "", max_tokens=120)
        assert len(out) > 1
        for chunk in out:
            assert chunk["text"].startswith("Doc > H1 > H2")
            assert chunk["heading"] == "H1 > H2"

    def test_all_chunks_within_cap(self):
        body = "\n\n".join(["word " * 60] * 8)
        raw = [{"text": body, "heading": "H1"}]
        for chunk in compose_chunks(raw, "Doc", "", max_tokens=120):
            assert count_tokens(chunk["text"]) <= 120

    def test_description_used_only_without_heading(self):
        with_heading = compose_chunks([{"text": "B.", "heading": "H1"}], "Doc", "Desc.")
        assert "Desc." not in with_heading[0]["text"]

        without = compose_chunks([{"text": "B.", "heading": ""}], "Doc", "Desc.")
        assert "Desc." in without[0]["text"]

    def test_realistic_file_respects_default_cap(self):
        content = "# Title\n\n" + "\n\n".join(["Some sentence here. " * 30] * 40)
        out = compose_chunks(split_by_headings(content), "Title", "A description.")
        assert out
        for chunk in out:
            assert count_tokens(chunk["text"]) <= MAX_TOKENS


class TestInferProductArea:
    def test_nested_file_uses_top_level_dir(self):
        path = _DOCS_ROOT / "markdown" / "now-platform" / "sub" / "page.md"
        assert infer_product_area(path) == "now-platform"

    def test_file_directly_under_markdown_has_no_area(self):
        # parts[0] would be the filename here, which is not a product area.
        assert infer_product_area(_DOCS_ROOT / "markdown" / "stray.md") == "unknown"


class TestParseFrontmatterLoosely:
    def test_extracts_scalars_and_body(self):
        meta, body = parse_frontmatter_loosely("---\ntitle: X\nlocale: en-US\n---\n\nBody.")
        assert meta == {"title": "X", "locale": "en-US"}
        assert body == "Body."

    def test_recovers_title_containing_a_colon(self):
        # The exact upstream shape strict YAML rejects, e.g.
        # 'title: Example: WS-Security SOAP envelope header'
        meta, body = parse_frontmatter_loosely(
            "---\ntitle: Example: WS-Security SOAP envelope header\n---\n\nBody."
        )
        assert meta["title"] == "Example: WS-Security SOAP envelope header"
        assert body == "Body."

    def test_skips_lists_and_nested_values(self):
        meta, _ = parse_frontmatter_loosely(
            "---\ntitle: X\nkeywords: [a, b]\nbreadcrumb:\n  - One\n  - Two\n---\n\nB."
        )
        assert meta["title"] == "X"
        assert "keywords" not in meta and "breadcrumb" not in meta

    def test_strips_quotes(self):
        meta, _ = parse_frontmatter_loosely('---\nlast_updated: "2026-03-12"\n---\n\nB.')
        assert meta["last_updated"] == "2026-03-12"

    def test_plain_text_untouched(self):
        meta, body = parse_frontmatter_loosely("# Heading\n\nBody.")
        assert meta == {}
        assert body == "# Heading\n\nBody."


class TestMakeChunkId:
    def test_deterministic(self):
        assert make_chunk_id(_FAKE_PATH, 0) == make_chunk_id(_FAKE_PATH, 0)

    def test_different_index_gives_different_id(self):
        assert make_chunk_id(_FAKE_PATH, 0) != make_chunk_id(_FAKE_PATH, 1)

    def test_different_path_gives_different_id(self):
        other = _DOCS_ROOT / "markdown" / "other_file.md"
        assert make_chunk_id(_FAKE_PATH, 0) != make_chunk_id(other, 0)

    def test_returns_32_char_hex_string(self):
        result = make_chunk_id(_FAKE_PATH, 0)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)
