from pathlib import Path
from use_terminal.color import Rainbow, color
from pars.pars_content import Parser
from pars.finder import Finder


def main() -> None:
    parser = Parser(Path("."))
    source_files = parser.parse_repository()

    all_chunks = [
        chunk
        for source_file in source_files.root
        for chunk in source_file.chunks
    ]
    rainbow = Rainbow(degrees=40)
    finder = Finder()
    finder.build_index(all_chunks)

    results = finder.search("ou son les token", top_k=5)

    for chunk in results:
        bloup = rainbow.give()
        print(f"File: {chunk.source_path}")
        print(f"Characters: {chunk.start_char}-{chunk.end_char}")
        print(color(chunk.content, bloup[0], bloup[1], bloup[2]))
        print("---")


if __name__ == "__main__":
    main()


# from pathlib import Path


# path = Path("README.md")
# content = path.read_text(encoding="utf-8")

# chunker = Chunker(max_chunk_size=2000)
# chunks = chunker.chunk(path, content)

# for chunk in chunks:
#    assert len(chunk.content) <= 2000
#    assert content[chunk.start_char : chunk.end_char] == chunk.content

# print(f"Nombre de chunks : {len(chunks)}")
# print("Tous les chunks respectent la limite et les positions.")
