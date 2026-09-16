import json

import pytest

from recsys_platform.evaluation import EvaluationResult


@pytest.fixture(scope="module")
def evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        {
            5: {"precision": 0.25, "recall": 0.5, "hit_rate": 0.75, "ndcg": 0.6},
            10: {"precision": 0.2, "recall": 0.7, "hit_rate": 1.0, "ndcg": 0.8},
        }
    )


def test_result_behaves_like_mapping(evaluation_result) -> None:
    assert evaluation_result[5]["recall"] == 0.5
    assert list(evaluation_result) == [5, 10]


def test_result_save_json(evaluation_result, tmp_path) -> None:
    path = tmp_path / "result.json"
    evaluation_result.save(path)

    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["5"]["precision"] == 0.25
    assert saved["10"]["ndcg"] == 0.8


def test_result_save_str(evaluation_result, tmp_path) -> None:
    path = tmp_path / "result.txt"
    evaluation_result.save(path)

    output = path.read_text(encoding="utf-8")

    assert "Precision" in output
    assert "Recall" in output
    assert "Hit Rate" in output
    assert "Ndcg" in output
