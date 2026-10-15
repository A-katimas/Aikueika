from pathlib import Path
from use_terminal.color import chose_color

from pars.pars_content import Parser


def main() -> None:
    parser = Parser(Path("Rag_subject"))
    files = parser.parse_repository()

    print(f"Files parsed: {len(files.root)}")
    i = 0
    for source_file in files.root[:20]:

        source = source_file.path
        source2 = source_file.extension
        print(chose_color((source, source2), i))
        i += 1


if __name__ == "__main__":
    main()
