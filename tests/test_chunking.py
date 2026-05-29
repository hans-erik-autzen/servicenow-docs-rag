import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from chunk_and_index import _REPO_ROOT, make_chunk_id, split_by_headings, split_on_blank_lines

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
        # No blank lines to split on — must be returned whole even if over max_tokens
        text = "word " * 500
        result = split_on_blank_lines(text, max_tokens=10)
        assert len(result) == 1
        assert "word" in result[0]


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
