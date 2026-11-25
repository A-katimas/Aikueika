"""Compute recall@k of search results against a ground-truth dataset."""

from pathlib import Path

from pydantic import BaseModel

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluation(BaseModel):
    """Result of an evaluation."""

    recall: dict[int, float]
    questions: int
    without_results: int
    too_long_sources: int


class Evaluator:
    """Measure how well retrieved sources match the expected ones.

    A correct source counts as found when one retrieved source is in the
    same file and its character range overlaps it enough (IoU >= min_iou).
    The end of a range is not included, like in a Python slice.
    """

    def __init__(
        self,
        min_iou: float = 0.05,
        max_context_length: int = 2000,
    ) -> None:
        """Set the overlap threshold and the maximum source length."""
        self.min_iou = min_iou
        self.max_context_length = max_context_length

    @staticmethod
    def iou(first: MinimalSource, second: MinimalSource) -> float:
        """Return the intersection over union of two sources.

        Sources from different files never overlap.
        """
        if first.file_path != second.file_path:
            return 0.0
        start = max(first.first_character_index, second.first_character_index)
        end = min(first.last_character_index, second.last_character_index)
        intersection = max(0, end - start)
        union = (
            (first.last_character_index - first.first_character_index)
            + (second.last_character_index - second.first_character_index)
            - intersection
        )
        if union <= 0:
            return 0.0
        return intersection / union

    def is_found(
        self,
        expected: MinimalSource,
        retrieved: list[MinimalSource],
    ) -> bool:
        """Tell if an expected source is covered by a retrieved one."""
        return any(
            self.iou(expected, item) >= self.min_iou for item in retrieved
        )

    def recall_at_k(
        self,
        expected: list[MinimalSource],
        retrieved: list[MinimalSource],
        k: int,
    ) -> float:
        """Share of expected sources found in the first k results."""
        if not expected:
            return 0.0
        top = retrieved[:k]
        found = sum(1 for source in expected if self.is_found(source, top))
        return found / len(expected)

    @staticmethod
    def load_results(path: str) -> StudentSearchResults:
        """Read and validate a StudentSearchResults JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return StudentSearchResults.model_validate_json(text)

    @staticmethod
    def load_dataset(path: str) -> RagDataset:
        """Read and validate a ground-truth dataset file."""
        text = Path(path).read_text(encoding="utf-8")
        return RagDataset.model_validate_json(text)

    def evaluate(
        self,
        results: StudentSearchResults,
        dataset: RagDataset,
        ks: list[int],
    ) -> Evaluation:
        """Compute the mean recall@k over the ground-truth questions.

        Questions are matched by question_id, or by text if the id is
        unknown. A question without results counts as a recall of 0.
        """
        truths = [
            question
            for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion) and question.sources
        ]
        if not truths:
            raise ValueError("the dataset has no ground-truth sources")

        used_ks = [k for k in ks if 0 < k <= results.k]
        if not used_ks:
            raise ValueError(
                f"no valid k: the results were built with k={results.k}"
            )

        by_id = {item.question_id: item for item in results.search_results}
        by_text = {item.question: item for item in results.search_results}

        totals = {k: 0.0 for k in used_ks}
        without_results = 0
        for question in truths:
            found = by_id.get(question.question_id)
            if found is None:
                found = by_text.get(question.question)
            if found is None:
                without_results += 1
                continue
            for k in used_ks:
                totals[k] += self.recall_at_k(
                    question.sources,
                    found.retrieved_sources,
                    k,
                )

        too_long = sum(
            1
            for item in results.search_results
            for source in item.retrieved_sources
            if source.last_character_index - source.first_character_index
            > self.max_context_length
        )
        return Evaluation(
            recall={k: totals[k] / len(truths) for k in used_ks},
            questions=len(truths),
            without_results=without_results,
            too_long_sources=too_long,
        )
