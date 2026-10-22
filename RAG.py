from pathlib import Path
from use_terminal.color import Rainbow, color
from src.pars_content import Parser
from src.finder import Finder
from src.generator import Generator


def main() -> None:
    parser = Parser(Path("."))
    source_files = parser.parse_repository()

    generator = Generator()

    all_chunks = [
        chunk
        for source_file in source_files.root
        for chunk in source_file.chunks
    ]
    rainbow = Rainbow(degrees=40)
    question = "how to make a backfilp"
    finder = Finder()
    finder.build_index(all_chunks)

    results = finder.search(question, top_k=5)

    for chunk in results:
        bloup = rainbow.give()
        print(f"File: {chunk.source_path}")
        print(f"Characters: {chunk.start_char}-{chunk.end_char}")
        print(color(chunk.content, bloup[0], bloup[1], bloup[2]))
        print("---")

    rep = generator.generate(question, results)
    print(rep)


if __name__ == "__main__":
    main()
