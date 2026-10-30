import pytest

from recsys_platform.models.matrix_factorization import MatrixFactorizationRecommender
from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.serving.registry import ModelRegistry, create_default_registry


def test_default_registry_resolves_builtin_models() -> None:
    registry = create_default_registry()

    assert tuple(registry) == ("popularity", "matrix_factorization")
    assert registry["popularity"] is PopularityRecommender
    assert registry["matrix_factorization"] is MatrixFactorizationRecommender


def test_registry_raises_for_unknown_family() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        _ = registry["unknown"]
