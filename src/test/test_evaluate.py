
import pytest

from src.evaluate import Evaluator
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def make_source(
    file_path: str = "src/example.py",
    start: int = 0,
    end: int = 100,
) -> MinimalSource:
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


def make_answered_question(
    question_id: str = "q1",
    question: str = "What is Python?",
    sources: list[MinimalSource] | None = None,
) -> AnsweredQuestion:
    return AnsweredQuestion.model_construct(
        question_id=question_id,
        question=question,
        sources=sources if sources is not None else [make_source()],
    )


def make_search_result(
    question_id: str = "q1",
    question: str = "What is Python?",
    sources: list[MinimalSource] | None = None,
) -> MinimalSearchResults:
    return MinimalSearchResults.model_construct(
        question_id=question_id,
        question=question,
        retrieved_sources=sources if sources is not None else [],
    )


def make_dataset(*questions: AnsweredQuestion) -> RagDataset:
    return RagDataset.model_construct(rag_questions=list(questions))


def make_results(
    *items: MinimalSearchResults,
    k: int = 5,
) -> StudentSearchResults:
    return StudentSearchResults.model_construct(
        search_results=list(items),
        k=k,
    )


class TestIoU:
    def test_identical_sources(self) -> None:
        source = make_source()

        assert Evaluator.iou(source, source) == 1.0

    def test_partial_overlap(self) -> None:
        first = make_source(start=0, end=100)
        second = make_source(start=50, end=150)

        assert Evaluator.iou(first, second) == pytest.approx(1 / 3)

    def test_no_overlap(self) -> None:
        first = make_source(start=0, end=50)
        second = make_source(start=50, end=100)

        assert Evaluator.iou(first, second) == 0.0

    def test_different_files(self) -> None:
        first = make_source("src/one.py", 0, 100)
        second = make_source("src/two.py", 0, 100)

        assert Evaluator.iou(first, second) == 0.0

    def test_empty_ranges(self) -> None:
        first = make_source(start=10, end=10)
        second = make_source(start=10, end=10)

        assert Evaluator.iou(first, second) == 0.0


class TestIsFound:
    def test_source_is_found(self) -> None:
        evaluator = Evaluator(min_iou=0.05)
        expected = make_source(start=0, end=100)
        retrieved = [make_source(start=50, end=100)]

        assert evaluator.is_found(expected, retrieved)

    def test_source_is_not_found_when_iou_is_too_low(self) -> None:
        evaluator = Evaluator(min_iou=0.5)
        expected = make_source(start=0, end=100)
        retrieved = [make_source(start=90, end=100)]

        assert not evaluator.is_found(expected, retrieved)

    def test_source_is_not_found_in_empty_results(self) -> None:
        evaluator = Evaluator()
        expected = make_source()

        assert not evaluator.is_found(expected, [])


class TestRecallAtK:
    def test_all_sources_found(self) -> None:
        evaluator = Evaluator()
        expected = [
            make_source(start=0, end=100),
            make_source(start=200, end=300),
        ]
        retrieved = [
            make_source(start=0, end=100),
            make_source(start=200, end=300),
        ]

        assert evaluator.recall_at_k(expected, retrieved, 5) == 1.0

    def test_only_one_source_found(self) -> None:
        evaluator = Evaluator()
        expected = [
            make_source(start=0, end=100),
            make_source(start=200, end=300),
        ]
        retrieved = [make_source(start=0, end=100)]

        assert evaluator.recall_at_k(expected, retrieved, 5) == 0.5

    def test_k_limits_retrieved_sources(self) -> None:
        evaluator = Evaluator()
        expected = [make_source(start=200, end=300)]
        retrieved = [
            make_source(start=0, end=100),
            make_source(start=200, end=300),
        ]

        assert evaluator.recall_at_k(expected, retrieved, 1) == 0.0
        assert evaluator.recall_at_k(expected, retrieved, 2) == 1.0

    def test_empty_expected_sources(self) -> None:
        evaluator = Evaluator()

        assert evaluator.recall_at_k([], [make_source()], 5) == 0.0


class TestEvaluate:
    def test_perfect_recall(self) -> None:
        evaluator = Evaluator()
        expected = make_source()

        dataset = make_dataset(
            make_answered_question(sources=[expected])
        )
        results = make_results(
            make_search_result(sources=[expected]),
            k=5,
        )

        evaluation = evaluator.evaluate(dataset=dataset,
                                        results=results,
                                        ks=[1, 5])

        assert evaluation.recall == {1: 1.0, 5: 1.0}
        assert evaluation.questions == 1
        assert evaluation.without_results == 0
        assert evaluation.too_long_sources == 0

    def test_question_without_results_counts_as_zero(self) -> None:
        evaluator = Evaluator()
        dataset = make_dataset(make_answered_question())
        results = make_results(k=5)

        evaluation = evaluator.evaluate(results, dataset, ks=[5])

        assert evaluation.recall[5] == 0.0
        assert evaluation.without_results == 1
        assert evaluation.questions == 1

    def test_match_question_by_text_when_id_is_unknown(self) -> None:
        evaluator = Evaluator()
        expected = make_source()

        dataset = make_dataset(
            make_answered_question(
                question_id="dataset-id",
                question="What is Python?",
                sources=[expected],
            )
        )
        results = make_results(
            make_search_result(
                question_id="different-id",
                question="What is Python?",
                sources=[expected],
            ),
            k=5,
        )

        evaluation = evaluator.evaluate(results, dataset, ks=[5])

        assert evaluation.recall[5] == 1.0
        assert evaluation.without_results == 0

    def test_only_answered_questions_with_sources_are_evaluated(self) -> None:
        evaluator = Evaluator()
        dataset = make_dataset(
            make_answered_question(sources=[make_source()]),
            make_answered_question(
                question_id="q2",
                sources=[],
            ),
        )
        results = make_results(k=5)

        evaluation = evaluator.evaluate(results, dataset, [5])

        assert evaluation.questions == 1

    def test_count_sources_that_are_too_long(self) -> None:
        evaluator = Evaluator(max_context_length=2000)
        expected = make_source()

        dataset = make_dataset(make_answered_question(sources=[expected]))
        results = make_results(
            make_search_result(
                sources=[
                    make_source(start=0, end=2001),
                    make_source(start=0, end=2000),
                ]
            ),
            k=5,
        )

        evaluation = evaluator.evaluate(results, dataset, [5])

        assert evaluation.too_long_sources == 1

    def test_raises_error_when_dataset_has_no_ground_truth(self) -> None:
        evaluator = Evaluator()
        dataset = make_dataset()
        results = make_results(k=5)

        with pytest.raises(ValueError, match="no ground-truth sources"):
            evaluator.evaluate(results, dataset, [5])

    def test_raises_error_when_no_valid_k(self) -> None:
        evaluator = Evaluator()
        dataset = make_dataset(make_answered_question())
        results = make_results(k=5)

        with pytest.raises(ValueError, match="no valid k"):
            evaluator.evaluate(results, dataset, [0, 6])

    def test_ignores_invalid_k_values(self) -> None:
        evaluator = Evaluator()
        expected = make_source()

        dataset = make_dataset(make_answered_question(sources=[expected]))
        results = make_results(
            make_search_result(sources=[expected]),
            k=5,
        )

        evaluation = evaluator.evaluate(results, dataset, [0, 1, 10])

        assert evaluation.recall == {1: 1.0}
