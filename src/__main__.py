import sys

import fire

from src.evaluate import Evaluator
from src.pipeline import Pipeline
from src.use_terminal.color import Rainbow, color

RAW_DIR = "data/raw"
MAX_CHUNK_SIZE = 2000
ERRORS = (OSError, ValueError, TypeError)


def print_error(message: str) -> None:
    """Print an error message on stderr."""
    print(f"Error: {message}", file=sys.stderr)


class Cli:
    """Commands exposed through Python Fire.

    Each public method is a command:
    uv run python -m src <method_name> [options]
    """

    def __init__(self) -> None:
        self.rainbow = Rainbow(degrees=15)

    def index(
        self,
        raw_dir: str = RAW_DIR,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """Build the index of the corpus."""
        try:
            pipeline = Pipeline(raw_dir, max_chunk_size)
            count = pipeline.build_index()
        except ERRORS as error:
            print_error(str(error))
            return
        print(f"Indexed {count} chunks")

    def search(
        self,
        query: str,
        k: int = 5,
        raw_dir: str = RAW_DIR,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """Print the top-k sources for a single query."""
        if not str(query).strip():
            print_error("the query is empty")
            return
        try:
            pipeline = Pipeline(raw_dir, max_chunk_size)
            sources = pipeline.search(str(query), k)
        except ERRORS as error:
            print_error(str(error))
            return
        if not sources:
            print("No result found.")
            return
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        raw_dir: str = RAW_DIR,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """Search every question of a dataset and save a JSON file."""
        try:
            pipeline = Pipeline(raw_dir, max_chunk_size)
            dataset = pipeline.load_dataset(dataset_path)
            results = pipeline.search_dataset(dataset, k)
            output_file = pipeline.save_results(
                results,
                dataset_path,
                save_directory,
            )
        except ERRORS as error:
            print_error(str(error))
            return
        print(f"Saved student_search_results to {output_file}")

    def answer(
        self,
        query: str,
        k: int = 5,
        raw_dir: str = RAW_DIR,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """Answer a single query using the retrieved context.

        With --stream True the answer is printed while it is generated.
        """
        try:
            pipeline = Pipeline(raw_dir, max_chunk_size)
            result = pipeline.answer(str(query), k)
        except ERRORS as error:
            print_error(str(error))
            return
        print("Sources:")
        for source in result.retrieved_sources:
            wouf = self.rainbow.give()
            print(
                color(
                    f"  {source.file_path} "
                    f"[{source.first_character_index}:"
                    f"{source.last_character_index}]",
                    wouf[0],
                    wouf[1],
                    wouf[2],
                )
            )
        if result.retrieved_sources:
            print("Answer:")
            print(color(result.answer, 50, 200, 100))
        else:
            print(
                color("sorry we don't have the sources to know that", 99, 5, 5)
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        raw_dir: str = RAW_DIR,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """Generate answers from a search results file and save a JSON."""
        try:
            pipeline = Pipeline(raw_dir, max_chunk_size)
            search_results = pipeline.load_search_results(
                student_search_results_path
            )
            print(f"Loaded {len(search_results.search_results)} questions")
            answers = pipeline.answer_dataset(search_results)
            output_file = pipeline.save_results(
                answers,
                student_search_results_path,
                save_directory,
            )
        except ERRORS as error:
            print_error(str(error))
            return
        print(f"Saved student_search_results_and_answer to {output_file}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        ks: tuple[int, ...] | int = (1, 3, 5, 10),
        max_context_length: int = 2000,
    ) -> None:
        """Print the recall@k of search results against a ground truth."""
        k_list = [ks] if isinstance(ks, int) else list(ks)
        try:
            evaluator = Evaluator(max_context_length=max_context_length)
            results = evaluator.load_results(student_search_results_path)
            dataset = evaluator.load_dataset(dataset_path)
            report = evaluator.evaluate(results, dataset, k_list)
        except ERRORS as error:
            print_error(str(error))
            return
        print(f"Student data is valid: {report.too_long_sources == 0}")
        print("Evaluation Results")
        print("=" * 40)
        print(
            " ".join(
                f"Recall@{k}: {value:.3f}"
                for k, value in report.recall.items()
            )
        )
        print(
            f"Questions: {report.questions}, "
            f"without results: {report.without_results}, "
            f"sources too long: {report.too_long_sources}"
        )


if __name__ == "__main__":
    fire.Fire(Cli)
