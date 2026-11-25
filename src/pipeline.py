"""RAG pipeline: indexing, searching and answering."""

import uuid
from pathlib import Path

from tqdm import tqdm

from src.chunker import Chunk
from src.finder import Finder
from src.generator import Generator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.pars_content import Parser

NO_ANSWER = "No relevant information was found to answer this question."


class Pipeline:
    """Index a corpus once, then search it and answer questions.

    Errors (missing folder, invalid k, bad JSON...) are raised as
    OSError or ValueError so that the caller decides how to report them.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        max_chunk_size: int = 2000,
        max_context_passages: int = 3,
    ) -> None:
        """Prepare the pipeline without reading anything yet."""
        self.raw_dir = Path(raw_dir)
        self.max_chunk_size = max_chunk_size
        self.max_context_passages = max_context_passages
        self.finder = Finder()
        self.is_indexed = False
        self.generator: Generator | None = None
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")
        if max_chunk_size > 2000:
            raise ValueError("max chunk can be more 2000")

    # ------------------------------------------------------------ index

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

    # ----------------------------------------------------------- search

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
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
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
        if not isinstance(k, int) or k <= 0:
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

    # ----------------------------------------------------------- answer

    def get_generator(self) -> Generator:
        """Create the generator on first use (loading Qwen is slow)."""
        if self.generator is None:
            self.generator = Generator()
        return self.generator

    @staticmethod
    def read_passage(source: MinimalSource) -> str:
        """Read the text covered by a source from its file."""
        content = Path(source.file_path).read_text(
            encoding="utf-8",
            errors="replace",
        )
        start = source.first_character_index
        end = source.last_character_index
        return content[start:end]

    def answer_question(
        self,
        question_id: str,
        question: str,
        sources: list[MinimalSource],
    ) -> MinimalAnswer:
        """Generate an answer to a question from its retrieved sources.

        Only the first max_context_passages sources are given to the
        model (small token budget), but all sources are kept in the
        result.
        """
        passages: list[str] = []
        for source in sources[: self.max_context_passages]:
            try:
                text = self.read_passage(source)
            except OSError as e:
                print(e)
                continue
            if text.strip():
                passages.append(f"File: {source.file_path}\n{text}")

        if passages:
            answer = self.get_generator().generate(
                question,
                passages,
            )
        else:
            answer = NO_ANSWER

        return MinimalAnswer(
            question_id=question_id,
            question=question,
            retrieved_sources=sources,
            answer=answer,
        )

    def answer(
        self,
        query: str,
        k: int = 5,
    ) -> MinimalAnswer:
        """Search then answer a single question."""
        if not query.strip():
            raise ValueError("the query is empty")
        sources = self.search(query, k)
        return self.answer_question(
            str(uuid.uuid4()),
            query,
            sources,
        )

    @staticmethod
    def load_search_results(path: str) -> StudentSearchResults:
        """Read and validate a StudentSearchResults JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return StudentSearchResults.model_validate_json(text)

    def answer_dataset(
        self,
        results: StudentSearchResults,
    ) -> StudentSearchResultsAndAnswer:
        """Answer every question of a StudentSearchResults.

        No search is done here: the sources come from the input file.
        """
        answers: list[MinimalAnswer] = []
        for item in tqdm(results.search_results, desc="Answering"):
            answers.append(
                self.answer_question(
                    item.question_id,
                    item.question,
                    item.retrieved_sources,
                )
            )
        return StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k,
        )

    # ------------------------------------------------------------- save

    @staticmethod
    def save_results(
        results: StudentSearchResults | StudentSearchResultsAndAnswer,
        input_path: str,
        save_directory: str,
    ) -> Path:
        """Write the results as JSON, named after the input file."""
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / Path(input_path).name
        output_file.write_text(
            results.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return output_file
