import pytest
from pathlib import Path
from src.pars_content import Parser


def test_discover_supported_files(tmp_path: Path) -> None:
    """Find Python and Markdown files."""
    (tmp_path / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Project",
        encoding="utf-8",
    )
    (tmp_path / "image.png").write_text(
        "not an image",
        encoding="utf-8",
    )

    parser = Parser(tmp_path)

    files = parser.discover_files()

    assert files == sorted(
        [
            tmp_path / "README.md",
            tmp_path / "main.py",
        ]
    )


def test_ignored_directories_are_excluded(
    tmp_path: Path,
) -> None:
    """Ignore excluded directories."""
    ignored_dir = tmp_path / "__pycache__"
    ignored_dir.mkdir()
    (ignored_dir / "cached.py").write_text(
        "print('ignored')",
        encoding="utf-8",
    )

    (tmp_path / "main.py").write_text(
        "print('included')",
        encoding="utf-8",
    )

    parser = Parser(tmp_path)

    files = parser.discover_files()

    assert files == [tmp_path / "main.py"]


def test_parse_file_reads_content_and_chunks(
    tmp_path: Path,
) -> None:
    """Read a file and generate its chunks."""
    source = tmp_path / "README.md"
    content = "# Project\n\nA small test project."
    source.write_text(content, encoding="utf-8")

    parser = Parser(tmp_path)

    result = parser.parse_file(source)

    assert result is not None
    assert result.path == Path("README.md")
    assert result.extension == ".md"
    assert result.content == content
    assert result.size == len(content)
    assert result.chunks
    assert all(len(chunk.content) <= 2000 for chunk in result.chunks)


def test_invalid_root_raises_error(tmp_path: Path) -> None:
    """Raise an error if the root is not a directory."""
    missing_root = tmp_path / "does_not_exist"
    parser = Parser(missing_root)

    with pytest.raises(NotADirectoryError):
        parser.discover_files()


def test_parse_unsupported_extension_returns_none(
    tmp_path: Path,
) -> None:
    """Ignore unsupported files when parsing directly."""
    source = tmp_path / "image.png"
    source.write_text("not an image", encoding="utf-8")

    parser = Parser(tmp_path)

    assert parser.parse_file(source) is None
