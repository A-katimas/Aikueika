import pytest
from pathlib import Path
from pars.chunker import Chunker


def test_chunks_respect_max_size() -> None:
    """Check that no chunk exceeds the maximum size."""
    chunker = Chunker(max_chunk_size=25)
    content = "word " * 40

    chunks = chunker.chunk_markdown(
        Path("README.md"),
        content,
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 25 for chunk in chunks)


def test_chunk_offsets_match_content() -> None:
    """Check that chunk offsets point to the correct text."""
    chunker = Chunker(max_chunk_size=25)
    content = "word " * 40

    chunks = chunker.chunk_markdown(
        Path("README.md"),
        content,
    )

    for chunk in chunks:
        assert chunk.content == content[chunk.start_char: chunk.end_char]
        assert chunk.start_char < chunk.end_char


def test_empty_file_returns_no_chunks() -> None:
    """Check that empty files produce no chunks."""
    chunker = Chunker()

    assert chunker.chunk(Path("README.md"), "") == []
    assert chunker.chunk(Path("main.py"), "") == []


def test_unsupported_extension_returns_no_chunks() -> None:
    """Check that unsupported files produce no chunks."""
    chunker = Chunker()

    assert (
        chunker.chunk(
            Path("image.png"),
            "some content",
        )
        == []
    )


def test_max_chunk_size_must_be_positive() -> None:
    """Check that invalid maximum sizes raise an error."""
    with pytest.raises(ValueError):
        Chunker(max_chunk_size=0)


def test_invalid_python_uses_fallback() -> None:
    """Check that invalid Python is still split into chunks."""
    chunker = Chunker(max_chunk_size=20)
    content = "def broken(:\n    pass\n"

    chunks = chunker.chunk_python(
        Path("broken.py"),
        content,
    )

    assert chunks
    assert all(len(chunk.content) <= 20 for chunk in chunks)


def test_python_chunks_respect_max_size() -> None:
    """Check that Python chunks respect the size limit."""
    chunker = Chunker(max_chunk_size=40)
    content = "x = 1\n" "y = 2\n" "z = 3\n" "print(x + y + z)\n"

    chunks = chunker.chunk_python(
        Path("main.py"),
        content,
    )

    assert chunks
    assert all(len(chunk.content) <= 40 for chunk in chunks)


def test_long_markdown_paragraph_is_split() -> None:
    """Check that long Markdown content is split."""
    chunker = Chunker(max_chunk_size=20)
    content = "a" * 55

    chunks = chunker.chunk_markdown(
        Path("README.md"),
        content,
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content


def test_default_max_chunk_size_is_2000() -> None:
    """Check that the default chunk size is at most 2000 characters."""
    chunker = Chunker()
    content = "a" * 5000

    chunks = chunker.chunk_markdown(
        Path("README.md"),
        content,
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 2000 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content
