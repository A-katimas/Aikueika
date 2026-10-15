import ast
from pathlib import Path
from pydantic import BaseModel, Field, RootModel


class Chunk(BaseModel):
    """Represent a searchable fragment of a source file."""

    source_path: Path
    content: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class SourceFile(BaseModel):
    """Represent a source file."""

    path: Path
    extension: str
    content: str
    size: int = Field(ge=0)
    chunks: list[Chunk]


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
        chunk = self.chunking(path, content)
        return SourceFile(
            path=path.relative_to(self.root),
            extension=path.suffix,
            content=content,
            size=len(content),
            chunks=chunk,
        )

    def chunking(self, path: Path, content: str) -> list[Chunk]:
        """Split source content into chunks."""
        if path.suffix == ".md":
            return self.chunk_markdown(path, content)

        if path.suffix != ".py":
            return []

        """Split Python source into function chunks."""

        chunks: list[Chunk] = []
        tree = ast.parse(content)

        # Position de chaque début de ligne dans le texte
        line_starts = [0]
        for index, char in enumerate(content):
            if char == "\n":
                line_starts.append(index + 1)

        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_char = line_starts[func.lineno - 1]
                end_char = line_starts[func.end_lineno - 1]

                # Inclure toute la dernière ligne
                end_char = content.find("\n", end_char)
                if end_char == -1:
                    end_char = len(content)
                else:
                    end_char += 1

                chunk_content = content[start_char:end_char]

                chunks.append(
                    Chunk(
                        source_path=path,
                        content=chunk_content,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )

        return chunks

    def chunk_markdown(self, path: Path, content: str) -> list[Chunk]:
        """Split Markdown content into chunks."""
        # À implémenter : découpage par titres et paragraphes.
        return [
            Chunk(
                source_path=path,
                content=content,
                start_char=0,
                end_char=len(content),
            )
        ]

    def parse_repository(self) -> SourceFiles:
        """Discover and parse all supported repository files."""
        parsed_files = []

        for path in self.discover_files():
            source_file = self.parse_file(path)

            if source_file is not None:
                parsed_files.append(source_file)

        return SourceFiles(parsed_files)


def find_functions(path: Path) -> None:
    """Find function definitions in a Python file."""
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno

            print(f"Fonction : {node.name}")
            print(f"Début : ligne {start}, fin : ligne {end}")
            print()


# find_functions(Path("pars_content.py"))
