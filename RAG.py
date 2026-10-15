from pathlib import Path
from use_terminal.color import Rainbow, color

from pars.pars_content import Parser


def main() -> None:
    parser = Parser(Path("Rag_subject"))
    files = parser.parse_repository()

    print(f"Files parsed: {len(files.root)}")
    i = 0
    rainbow = Rainbow(degrees=30)
    for source_file in files.root[:50]:

        source = source_file.chunks[0].content
        source2 = source_file.extension
        bloup = rainbow.give()

        print(color((source), bloup[0], bloup[1], bloup[2]))
        i += 1


if __name__ == "__main__":
    main()
