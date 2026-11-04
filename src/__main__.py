from pathlib import Path
from src.pars_content import Parser
from src.finder import Finder
from src.generator import Generator
from src.Rag import Rag


def main() -> None:
    parser = Parser(Path("."))
    generator = Generator()
    finder = Finder()
    question = "How to configure OpenAI server?"

    rag = Rag(parser, finder, generator, 5, question)
    rag.print_chunk()


if __name__ == "__main__":
    main()
