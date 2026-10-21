# pars/chunker.py
import ast
import re
from pathlib import Path
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represent a searchable fragment of a source file."""

    source_path: Path
    content: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class Chunker:
    """Split source files into searchable chunks."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the chunker with a maximum chunk size."""
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")

        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        path: Path,
        content: str,
    ) -> list[Chunk]:
        """Choose a chunking strategy based on file extension."""
        if path.suffix == ".py":
            return self.chunk_python(path, content)

        if path.suffix == ".md":
            return self.chunk_markdown(path, content)

        return []

    def _split_content(
        self,
        path: Path,
        content: str,
        boundaries: list[int],
    ) -> list[Chunk]:
        """Split content using preferred boundaries and size limits."""
        chunks = []
        cursor = 0
        boundaries = sorted(set(boundaries + [len(content)]))

        while cursor < len(content):
            limit = min(
                cursor + self.max_chunk_size,
                len(content),
            )

            # Prefer a semantic boundary within the size limit.
            candidates = [
                boundary
                for boundary in boundaries
                if cursor < boundary <= limit
            ]

            if candidates:
                end = max(candidates)
            else:
                # No suitable semantic boundary: prefer a newline.
                newline = content.rfind("\n", cursor + 1, limit + 1)
                end = newline + 1 if newline >= cursor else limit

            # Guarantee forward progress.
            if end <= cursor:
                end = limit

            text = content[cursor:end]

            if text.strip():
                chunks.append(
                    Chunk(
                        source_path=path,
                        content=text,
                        start_char=cursor,
                        end_char=end,
                    )
                )

            cursor = end

        return chunks

    def chunk_python(
        self,
        path: Path,
        content: str,
    ) -> list[Chunk]:
        """Split Python code using top-level syntax boundaries."""
        if not content:
            return []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Still index files that cannot be parsed as valid Python.
            return self._split_content(
                path,
                content,
                self._line_boundaries(content),
            )

        line_starts = [0]
        for index, char in enumerate(content):
            if char == "\n":
                line_starts.append(index + 1)

        boundaries = []

        for node in tree.body:
            if node.end_lineno is None:
                continue

            end_line = node.end_lineno

            if end_line < len(line_starts):
                end = line_starts[end_line]
            else:
                end = len(content)

            boundaries.append(end)

        return self._split_content(path, content, boundaries)

    def chunk_markdown(
        self,
        path: Path,
        content: str,
    ) -> list[Chunk]:
        """Split Markdown using paragraph boundaries."""
        if not content:
            return []

        boundaries = [
            match.end() for match in re.finditer(r"\n[ \t]*\n+", content)
        ]

        return self._split_content(path, content, boundaries)

    def _line_boundaries(self, content: str) -> list[int]:
        """Return character offsets immediately after newlines."""
        return [
            index + 1 for index, char in enumerate(content) if char == "\n"
        ]
