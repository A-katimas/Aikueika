"""Search relevant source chunks with BM25."""

import re
from rank_bm25 import BM25Okapi
from pars.chunker import Chunk


class Finder:
    """Index chunks and retrieve the most relevant ones."""

    def __init__(self) -> None:
        """Initialize an empty search index."""
        self.chunks: list[Chunk] = []
        self.index: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into lowercase tokens."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from source chunks."""
        self.chunks = chunks

        if not chunks:
            self.index = None
            return

        tokenized_chunks = [self.tokenize(chunk.content) for chunk in chunks]
        self.index = BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Return the top-k chunks relevant to a query."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index is None:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.chunks[i] for i in ranked_indices[:top_k] if scores[i] > 0
        ]
