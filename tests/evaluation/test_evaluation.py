from math import log2

import pytest

from recsys_platform.evaluation import evaluate, evaluate_multiple, evaluation


@pytest.fixture
def ranking_case() -> tuple[list[int], list[int]]:
    """Ranking with three relevant items appearing at ranks 1, 3, and 5."""
    y_true = [10, 20, 30]
    y_pred = [10, 40, 20, 50, 30]

    return y_true, y_pred


def test_evaluate_returns_all_default_metrics(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate(y_true=y_true, y_pred=y_pred, k=3)
    assert set(result) == {"precision", "recall", "hit_rate", "ndcg"}


def test_evaluate_calculates_metrics_correctly(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate(y_true=y_true, y_pred=y_pred, k=3)

    expected_dcg = 1.0 + 1.0 / log2(4)
    expected_idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["ndcg"] == pytest.approx(expected_dcg / expected_idcg)


def test_evaluate_matches_single_result_from_evaluate_multiple(ranking_case) -> None:
    y_true, y_pred = ranking_case

    single = evaluate(y_true=y_true, y_pred=y_pred, k=3)
    multiple = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=(3,))

    assert single == multiple[3]


def test_evaluate_uses_default_k(ranking_case) -> None:
    y_true, y_pred = ranking_case

    default_result = evaluate(y_true=y_true, y_pred=y_pred)
    explicit_result = evaluate(y_true=y_true, y_pred=y_pred, k=10)

    assert default_result == explicit_result


def test_evaluate_selected_metrics_only(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate(y_true=y_true, y_pred=y_pred, k=3, metrics=("precision", "ndcg"))

    assert set(result) == {"precision", "ndcg"}
    assert result["precision"] == pytest.approx(2 / 3)


def test_evaluate_multiple_calculates_multiple_cutoffs(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=(1, 3, 5))

    # K = 1
    assert result[1]["precision"] == pytest.approx(1.0)
    assert result[1]["recall"] == pytest.approx(1 / 3)
    assert result[1]["hit_rate"] == pytest.approx(1.0)
    assert result[1]["ndcg"] == pytest.approx(1.0)

    # K = 3
    dcg_at_3 = 1.0 + 1.0 / log2(4)
    idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    assert result[3]["precision"] == pytest.approx(2 / 3)
    assert result[3]["recall"] == pytest.approx(2 / 3)
    assert result[3]["hit_rate"] == pytest.approx(1.0)
    assert result[3]["ndcg"] == pytest.approx(dcg_at_3 / idcg)

    # K = 5
    dcg_at_5 = 1.0 + 1.0 / log2(4) + 1.0 / log2(6)

    assert result[5]["precision"] == pytest.approx(3 / 5)
    assert result[5]["recall"] == pytest.approx(1.0)
    assert result[5]["hit_rate"] == pytest.approx(1.0)
    assert result[5]["ndcg"] == pytest.approx(dcg_at_5 / idcg)


def test_evaluate_multiple_sorts_and_deduplicates_k_values(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=(5, 1, 3, 5, 1))

    assert list(result) == [1, 3, 5]


def test_evaluate_multiple_deduplicates_metrics(ranking_case) -> None:
    y_true, y_pred = ranking_case

    result = evaluate_multiple(
        y_true=y_true, y_pred=y_pred, k_values=(3,), metrics=("precision", "recall", "precision")
    )

    assert list(result[3]) == ["precision", "recall"]


def test_empty_ground_truth_returns_zero_metrics() -> None:
    result = evaluate(y_true=[], y_pred=[10, 20, 30], k=3)

    assert result == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "ndcg": 0.0}


def test_empty_predictions_return_zero_metrics() -> None:
    result = evaluate(y_true=[10, 20], y_pred=[], k=3)

    assert result == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "ndcg": 0.0}


def test_precision_penalizes_short_prediction_list() -> None:
    result = evaluate(y_true=[10, 20], y_pred=[10], k=5)

    assert result["precision"] == pytest.approx(1 / 5)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["hit_rate"] == pytest.approx(1.0)

    expected_idcg = 1.0 + 1.0 / log2(3)

    assert result["ndcg"] == pytest.approx(1.0 / expected_idcg)


def test_predictions_beyond_k_do_not_affect_result() -> None:
    y_true = [20]

    result_with_later_hit = evaluate(y_true=y_true, y_pred=[10, 30, 20], k=2)
    result_without_hit = evaluate(y_true=y_true, y_pred=[10, 30], k=2)

    assert result_with_later_hit == result_without_hit
    assert result_with_later_hit == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "ndcg": 0.0}


def test_ndcg_rewards_higher_ranked_relevant_items() -> None:
    y_true = [10]

    high_rank = evaluate(y_true=y_true, y_pred=[10, 20, 30], k=3)
    low_rank = evaluate(y_true=y_true, y_pred=[20, 30, 10], k=3)

    assert high_rank["precision"] == low_rank["precision"]
    assert high_rank["recall"] == low_rank["recall"]
    assert high_rank["hit_rate"] == low_rank["hit_rate"]

    assert high_rank["ndcg"] == pytest.approx(1.0)
    assert low_rank["ndcg"] == pytest.approx(1.0 / log2(4))

    assert high_rank["ndcg"] > low_rank["ndcg"]


def test_duplicate_ground_truth_items_are_treated_as_single_relevance() -> None:
    duplicated = evaluate(y_true=[10, 10, 20], y_pred=[10, 30, 20], k=3)
    unique = evaluate(y_true=[10, 20], y_pred=[10, 30, 20], k=3)

    assert duplicated == unique


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 1 / 3)])
def test_hit_entering_top_k_changes_metrics(k: int, expected: float) -> None:
    result = evaluate(y_true=[20], y_pred=[10, 20, 30], k=k, metrics=("precision",))

    assert result["precision"] == pytest.approx(expected)


def test_custom_metric_uses_fallback_path(monkeypatch) -> None:
    calls: list[int] = []

    def custom_metric(y_true, y_pred, k) -> float:
        calls.append(k)

        predicted = set(y_pred[:k])
        return float(bool(predicted & set(y_true)))

    monkeypatch.setattr(evaluation.metrics_module, "custom_metric", custom_metric, raising=False)

    result = evaluate_multiple(
        y_true=[20], y_pred=[10, 20, 30], k_values=(1, 2, 3), metrics=("custom_metric",)
    )

    assert result == {
        1: {"custom_metric": 0.0},
        2: {"custom_metric": 1.0},
        3: {"custom_metric": 1.0},
    }

    assert calls == [1, 2, 3]


def test_mixed_builtin_and_custom_metrics_use_fallback(monkeypatch) -> None:
    def custom_metric(y_true, y_pred, k) -> float:
        return float(len(y_pred[:k]))

    monkeypatch.setattr(evaluation.metrics_module, "custom_metric", custom_metric, raising=False)
    result = evaluate_multiple(
        y_true=[10], y_pred=[10, 20, 30], k_values=(1, 3), metrics=("precision", "custom_metric")
    )

    assert result[1]["precision"] == pytest.approx(1.0)
    assert result[1]["custom_metric"] == pytest.approx(1.0)

    assert result[3]["precision"] == pytest.approx(1 / 3)
    assert result[3]["custom_metric"] == pytest.approx(3.0)


def test_builtin_metrics_use_optimized_path(monkeypatch, ranking_case) -> None:
    """Built-in multi-K evaluation should not call individual metric functions."""

    def fail(*args, **kwargs):
        pytest.fail("Individual metric function was called.")

    monkeypatch.setattr(evaluation.metrics_module, "precision", fail)
    monkeypatch.setattr(evaluation.metrics_module, "recall", fail)
    monkeypatch.setattr(evaluation.metrics_module, "hit_rate", fail)
    monkeypatch.setattr(evaluation.metrics_module, "ndcg", fail)

    y_true, y_pred = ranking_case

    result = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=(1, 3, 5))

    assert result[1]["precision"] == pytest.approx(1.0)
    assert result[3]["recall"] == pytest.approx(2 / 3)
    assert result[5]["recall"] == pytest.approx(1.0)
