from use_terminal.color import Rainbow, color
from src.pars_content import Parser
from src.finder import Finder
from src.generator import Generator
from src.chunker import Chunk


class Rag:
    def __init__(
        self,
        parser: Parser,
        finder: Finder,
        generator: Generator,
        top_k: int,
        question: str,
    ):
        self.top_k = top_k
        self.question = question
        self.parser = parser
        self.finder = finder
        self.generator = generator
        self.rainbow = Rainbow(degrees=40)
        self.source_files = self.parser.parse_repository()
        self.all_chunk = self.all_chunk_maker()
        finder.build_index(self.all_chunk)
        self.results = self.finder.search(self.question, top_k=self.top_k)
        self.repgene = self.generator.generate(self.question, self.results)

    def all_chunk_maker(self) -> list[Chunk]:
        all_chunks = [
            chunk
            for source_file in self.source_files.root
            for chunk in source_file.chunks
        ]
        return all_chunks

    def print_chunk(self) -> None:
        print("____chunking____")
        for chunk in self.results:
            bloup = self.rainbow.give()
            print(f"File: {chunk.source_path}")
            print(f"Characters: {chunk.start_char}-{chunk.end_char}")
            print(color(chunk.content, bloup[0], bloup[1], bloup[2]))
            print("---")
        print("____end chunk____")

    def print_generator(self) -> None:
        print(self.repgene)
