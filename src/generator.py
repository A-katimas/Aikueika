import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from src.chunker import Chunk
from use_terminal.color import Rainbow, color

MODEL_NAME = "Qwen/Qwen3-0.6B"


class RainbowStreamer(TextStreamer):
    def __init__(self, tokenizer, rainbow, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.rainbow = rainbow

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if text:
            bloup = self.rainbow.give()
            print(
                color(text, bloup[0], bloup[1], bloup[2]),
                end="",
                flush=True,
            )

        if stream_end:
            print()


class Generator:
    def __init__(self, token_used: int = 100) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )
        self.token_used = token_used
        self.model.eval()
        self.rainbow = Rainbow(degrees=5)

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"Fichier : {chunk.source_path}\n" f"Contenu :\n{chunk.content}"
            for chunk in chunks
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui répond aux questions sur du code. "
                    "Utilise uniquement le contexte fourni. "
                    "Si le contexte ne contient pas la réponse, dis-le clairement."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Contexte :\n{context}\n\n" f"Question : {question}"
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )

        # with torch.no_grad():
        #    outputs = self.model.generate(
        #        **inputs,
        #        max_new_tokens=self.token_used,
        #        do_sample=False,
        #    )

        streamer = RainbowStreamer(
            self.tokenizer,
            self.rainbow,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.token_used,
                do_sample=False,
                streamer=streamer,
            )

        # On retire les tokens correspondant au prompt.
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()
