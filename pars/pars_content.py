from pathlib import Path

from pydantic import BaseModel, Field, RootModel


class SourceFile(BaseModel):
    """Represent a source file."""

    path: Path
    extension: str
    content: str
    size: int = Field(ge=0)


class SourceFiles(RootModel[list[SourceFile]]):
    """Represent a collection of source files."""


class Parser:
    """Discover and parse source files in a repository."""

    SUPPORTED_EXTENSIONS = {".py", ".md"}

    IGNORED_DIRECTORIES = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
    }

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def discover_files(self) -> list[Path]:
        """Find supported files recursively."""
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")

        files = []

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            relative_path = path.relative_to(self.root)

            if any(
                part in self.IGNORED_DIRECTORIES
                for part in relative_path.parts[:-1]
            ):
                continue

            if path.suffix in self.SUPPORTED_EXTENSIONS:
                files.append(path)

        return sorted(files)

    def parse_file(self, path: Path) -> SourceFile | None:
        """Read a supported file into a SourceFile model."""
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            return None

        content = path.read_text(encoding="utf-8")

        return SourceFile(
            path=path.relative_to(self.root),
            extension=path.suffix,
            content=content,
            size=len(content),
        )

    def parse_repository(self) -> SourceFiles:
        """Discover and parse all supported repository files."""
        parsed_files = []

        for path in self.discover_files():
            source_file = self.parse_file(path)

            if source_file is not None:
                parsed_files.append(source_file)

        return SourceFiles(parsed_files)
