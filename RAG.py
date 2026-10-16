from pathlib import Path
from use_terminal.color import Rainbow, color
from pars.chunker import Chunker
from pars.pars_content import Parser


def main() -> None:
    parser = Parser(Path("Rag_subject"))
    files = parser.parse_repository()

    print(f"Files parsed: {len(files.root)}")
    i = 0
    rainbow = Rainbow(degrees=40)
    for source_file in files.root:
        if len(source_file.chunks) > 1:
            source = source_file.chunks[1].content
            source2 = source_file.extension
            bloup = rainbow.give()

            print(color((source), bloup[0], bloup[1], bloup[2]))
        i += 1


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
