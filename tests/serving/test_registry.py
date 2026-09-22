import pytest

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.serving.registry import ModelRegistry, create_default_registry


def test_default_registry_resolves_popularity_model() -> None:
    registry = create_default_registry()

    assert len(registry) == 1
    assert tuple(registry) == ("popularity",)
    assert registry["popularity"] is PopularityRecommender


def test_registry_raises_for_unknown_family() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        _ = registry["unknown"]
