"""Answer generation with a small local language model (Qwen3)."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM "
    "codebase and documentation. "
    "Use only the context provided by the user. "
    "If the context does not contain the answer, say so clearly. "
    "Answer concisely and do not invent anything."
    "Answer in at most two sentences."
)


class Generator:
    """Generate an answer from retrieved passages using Qwen3.

    The model is loaded on the first call to generate(), so commands
    that never answer (index, search...) do not pay the loading time.
    """

    def __init__(self, max_new_tokens: int = 256) -> None:
        """Prepare the generator without loading the model yet."""
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the tokenizer and the model, only once."""
        if self.model is not None:
            return
        load = AutoTokenizer.from_pretrained
        tok = load(MODEL_NAME)  # type: ignore[no-untyped-call, unused-ignore]
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )
        model.eval()  # type: ignore[no-untyped-call, unused-ignore]
        self.tokenizer = tok
        self.model = model

    def generate(self, question: str, passages: list[str]) -> str:
        """Answer a question from the given passages.

        Args:
            question: The question to answer.
            passages: Texts retrieved for this question, best first.

        Returns:
            The generated answer.
        """
        self.load()
        context = "\n\n".join(passages)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
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
        print(inputs["input_ids"].shape[1], "prompt tokens")
        import time

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        duree = round(time.time() - start, 1)
        print(
            duree,
            "s,",
            outputs.shape[1] - inputs["input_ids"].shape[1],
            "new tokens",
        )
        # Remove the tokens that belong to the prompt.
        prompt_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True,
        )
        return str(answer).strip()
