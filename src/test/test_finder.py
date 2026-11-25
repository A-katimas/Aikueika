import pytest
from pathlib import Path
from src.chunker import Chunk
from src.finder import Finder


def test_finder_finds_relevant_chunk() -> None:
    """Check that Finder retrieves the relevant chunk."""

    config_chunk = Chunk(
        source_path=Path("src/config.py"),
        content="load configuration from yaml file",
        start_char=0,
        end_char=len("load configuration from yaml file"),
    )

    server_chunk = Chunk(
        source_path=Path("src/server.py"),
        content="start the http server",
        start_char=0,
        end_char=len("start the http server"),
    )

    chunks = [config_chunk, server_chunk]

    finder = Finder()
    finder.build_index(chunks)

    results = finder.search("configuration", top_k=2)

    assert len(results) > 0
    assert results[0].source_path == Path("src/config.py")
    assert results[0].content == "load configuration from yaml file"


def test_finder_returns_empty_list_without_index() -> None:
    """Check that Finder returns no results when the index is empty."""

    finder = Finder()

    results = finder.search("configuration")

    assert results == []


def test_finder_returns_empty_list_for_empty_query() -> None:
    """Check that Finder returns no results for an empty query."""

    chunk = Chunk(
        source_path=Path("src/config.py"),
        content="load configuration from yaml file",
        start_char=0,
        end_char=len("load configuration from yaml file"),
    )

    finder = Finder()
    finder.build_index([chunk])

    results = finder.search("")

    assert results == []


def test_finder_returns_empty_list_for_punctuation() -> None:
    """Check that punctuation-only queries return no results."""
    chunk = Chunk(
        source_path=Path("src/config.py"),
        content="load configuration from yaml file",
        start_char=0,
        end_char=len("load configuration from yaml file"),
    )

    finder = Finder()
    finder.build_index([chunk])

    results = finder.search("!!! ???")

    assert results == []


def test_finder_rejects_non_positive_top_k() -> None:
    """Check that non-positive top_k values raise an error."""
    finder = Finder()

    with pytest.raises(ValueError):
        finder.search("configuration", top_k=0)


def test_finder_returns_at_most_top_k_results() -> None:
    """Check that search respects top_k."""
    chunks = [
        Chunk(
            source_path=Path("src/config.py"),
            content="load configuration from yaml file",
            start_char=0,
            end_char=len("load configuration from yaml file"),
        ),
        Chunk(
            source_path=Path("src/server.py"),
            content="start the http server",
            start_char=0,
            end_char=len("start the http server"),
        ),
    ]

    finder = Finder()
    finder.build_index(chunks)

    results = finder.search("server", top_k=1)

    assert len(results) <= 1
