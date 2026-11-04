from pathlib import Path

from tqdm import tqdm

from src.chunker import Chunk
from src.finder import Finder
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.pars_content import Parser


class Pipeline:
    """Index a corpus once, then answer search requests.

    Errors (missing folder, invalid k, bad JSON...) are raised as
    OSError or ValueError so that the caller decides how to report them.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        max_chunk_size: int = 2000,
    ) -> None:
        """Prepare the pipeline without reading anything yet."""
        self.raw_dir = Path(raw_dir)
        self.max_chunk_size = max_chunk_size
        self.finder = Finder()
        self.is_indexed = False

    def build_index(self) -> int:
        """Parse and chunk the corpus, then build the search index.

        Returns:
            The number of chunks indexed.
        """
        parser = Parser(self.raw_dir, self.max_chunk_size)
        source_files = parser.parse_repository()
        chunks = [
            chunk
            for source_file in source_files.root
            for chunk in source_file.chunks
        ]
        if not chunks:
            raise ValueError(f"no .py or .md file found in {self.raw_dir}")

        self.finder.build_index(chunks)
        self.is_indexed = True
        return len(chunks)

    @staticmethod
    def to_source(chunk: Chunk) -> MinimalSource:
        """Convert a Chunk into the MinimalSource expected by the grader."""
        return MinimalSource(
            file_path=chunk.source_path.as_posix(),
            first_character_index=chunk.start_char,
            last_character_index=chunk.end_char,
        )

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Return the top-k sources for one query.

        The index is built on the first call if needed.
        """
        if not self.is_indexed:
            self.build_index()
        chunks = self.finder.search(query, top_k=k)
        return [self.to_source(chunk) for chunk in chunks]

    @staticmethod
    def load_dataset(dataset_path: str) -> RagDataset:
        """Read and validate a dataset file."""
        text = Path(dataset_path).read_text(encoding="utf-8")
        return RagDataset.model_validate_json(text)

    def search_dataset(
        self,
        dataset: RagDataset,
        k: int = 10,
    ) -> StudentSearchResults:
        """Search every question of a dataset."""
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if not self.is_indexed:
            self.build_index()

        results: list[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching"):
            results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=self.search(item.question, k),
                )
            )
        return StudentSearchResults(search_results=results, k=k)

    @staticmethod
    def save_results(
        results: StudentSearchResults,
        dataset_path: str,
        save_directory: str,
    ) -> Path:
        """Write the results as JSON, named after the dataset file."""
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / Path(dataset_path).name
        output_file.write_text(
            results.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return output_file
