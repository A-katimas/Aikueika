# from pathlib import Path
# from src.pars_content import Parser
# from src.finder import Finder
# from src.generator import Generator
# from src.Rag import Rag

# def main() -> None:
#    parser = Parser(Path("."))
#    generator = Generator()
#    finder = Finder()
#    question = "How to configure OpenAI server?"

#    rag = Rag(parser, finder, generator, 5, question)
#    rag.print_chunk()
#    print("fin")


# if __name__ == "__main__":
#    main()

import sys

import fire

from src.pipeline import Pipeline

RAW_DIR = "data/raw"
MAX_CHUNK_SIZE = 2000


def print_error(message: str) -> None:
    """Print an error message on stderr."""
    print(f"Error: {message}", file=sys.stderr)


class Cli:
    """Commands exposed through Python Fire.

    Each public method is a command:
    uv run python -m src <method_name> [options]
    """

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
        except (OSError, ValueError) as error:
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

        except (OSError, ValueError) as error:
            print_error(str(error))
            return
        print(f"Saved student_search_results to {output_file}")


if __name__ == "__main__":
    fire.Fire(Cli)
