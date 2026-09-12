import numpy as np
import pytest

from recsys_platform.evaluation.metrics import hit_rate, ndcg, precision, recall


class TestHitRateAtK:
    def test_hit(self) -> None:
        assert hit_rate(y_pred=[10, 20, 30], y_true={20, 40}, k=3) == 1.0

    def test_miss(self) -> None:
        assert hit_rate(y_pred=[10, 20, 30], y_true={40, 50}, k=3) == 0.0

    def test_respects_k(self) -> None:
        predictions = [10, 20, 30]

        assert hit_rate(y_pred=predictions, y_true={30}, k=2) == 0.0
        assert hit_rate(y_pred=predictions, y_true={30}, k=3) == 1.0

    def test_empty_y_true(self) -> None:
        assert hit_rate(y_pred=[10, 20], y_true=set(), k=2) == 0.0


class TestRecallAtK:
    def test_all_y_true_items_retrieved(self) -> None:
        assert recall(y_pred=[10, 20, 30], y_true={10, 20}, k=3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall(y_pred=[10, 20, 30, 40], y_true={20, 40, 50}, k=4) == pytest.approx(2 / 3)

    def test_no_y_true_items_retrieved(self) -> None:
        assert recall(y_pred=[10, 20, 30], y_true={40, 50}, k=3) == 0.0

    def test_respects_k(self) -> None:
        assert recall(y_pred=[10, 20, 30], y_true={20, 30}, k=2) == pytest.approx(0.5)

    def test_empty_y_true(self) -> None:
        assert recall(y_pred=[10, 20], y_true=set(), k=2) == 0.0

    def test_duplicate_y_true_items_do_not_change_score(self) -> None:
        assert recall(y_pred=[10, 20], y_true=[10, 10, 20], k=2) == 1.0


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        assert ndcg(
            y_pred=[10, 20, 30],
            y_true={10, 20},
            k=3,
        ) == pytest.approx(1.0)

    def test_y_true_item_is_discounted_by_rank(self) -> None:
        top = ndcg(y_pred=[10, 20, 30], y_true={10}, k=3)

        bottom = ndcg(y_pred=[20, 30, 10], y_true={10}, k=3)

        assert top == pytest.approx(1.0)
        assert bottom == pytest.approx(0.5)
        assert top > bottom

    def test_partial_ranking(self) -> None:
        score = ndcg(y_pred=[10, 20, 30], y_true={20, 30}, k=3)

        expected_dcg = 1 / np.log2(3) + 1 / np.log2(4)
        expected_idcg = 1 + 1 / np.log2(3)

        assert score == pytest.approx(expected_dcg / expected_idcg)

    def test_no_y_true_items_retrieved(self) -> None:
        assert ndcg(y_pred=[10, 20, 30], y_true={40}, k=3) == 0.0

    def test_empty_y_true(self) -> None:
        assert ndcg(y_pred=[10, 20], y_true=set(), k=2) == 0.0

    def test_respects_k(self) -> None:
        assert ndcg(y_pred=[20, 10], y_true={10}, k=1) == 0.0


@pytest.mark.parametrize("metric", [recall, ndcg])
def test_metric_rejects_non_positive_k(metric) -> None:
    with pytest.raises(ValueError, match="k must be greater than 0"):
        metric(y_pred=[10], y_true={10}, k=0)


class TestPrecisionAtK:
    def test_partial_precision(self) -> None:
        assert precision(y_pred=[10, 20, 30, 40, 50], y_true={20, 40, 60}, k=5) == pytest.approx(
            2 / 5
        )

    def test_perfect_precision(self) -> None:
        assert precision(y_pred=[10, 20], y_true={10, 20, 30}, k=2) == 1.0

    def test_zero_precision(self) -> None:
        assert precision(y_pred=[10, 20], y_true={30, 40}, k=2) == 0.0

    def test_respects_k(self) -> None:
        assert precision(y_pred=[10, 20, 30], y_true={30}, k=2) == 0.0

    def test_short_prediction_list_is_penalized(self) -> None:
        assert precision(y_pred=[10, 20], y_true={10, 20}, k=5) == pytest.approx(2 / 5)
