import os
from typing import Annotated

import numpy as np
from pydantic import Field
from scipy.sparse import csr_matrix

from recsys_platform.data import RecommenderDataset
from recsys_platform.training import Trainer, TrainerConfig
from recsys_platform.types import Float32Vector, UInt32Vector

from .model import MatrixFactorizationRecommender


class MatrixFactorizationTrainerConfig(TrainerConfig):
    fallback_items: Annotated[int, Field(gt=0)] = 20
    random_state: int | None = None
    n_jobs: int = -1

    @property
    def num_threads(self) -> int:
        return self._n_jobs_to_num_threads(self.n_jobs)

    @staticmethod
    def _n_jobs_to_num_threads(n_jobs: int) -> int:
        """
        Translate scikit-style ``n_jobs`` to implicit's ``num_threads``.

        ``n_jobs=-1`` uses all available CPUs, while implicit represents
        this as ``num_threads=0``.
        """
        if n_jobs == -1:
            return 0

        if n_jobs < -1:
            cpu_count = os.process_cpu_count() or 1
            return max(cpu_count + 1 + n_jobs, 1)

        return n_jobs


class MatrixFactorizationTrainer[ConfigType: MatrixFactorizationTrainerConfig](
    Trainer[MatrixFactorizationRecommender, ConfigType]
):
    """Base trainer for matrix-factorization recommenders."""

    def _get_fallback_predictions(
        self, user_items: csr_matrix, item_ids: UInt32Vector, max_items: int = 20
    ) -> tuple[UInt32Vector, Float32Vector]:
        counts = np.asarray(user_items.sum(axis=0)).ravel().astype(np.float32, copy=False)

        order = np.lexsort((item_ids, -counts))

        ranked_items = item_ids[order][:max_items]
        ranked_scores = counts[order][:max_items] / counts.max()

        return (ranked_items, ranked_scores.astype(np.float32, copy=False))

    def _build_user_item_matrix(
        self, data: RecommenderDataset
    ) -> tuple[csr_matrix, UInt32Vector, UInt32Vector]:
        interactions = (
            data.data.group_by("user_id", "item_id")
            .len(name="interaction_count")
            .collect(engine="streaming")
        )

        user_ids, user_indices = np.unique(interactions["user_id"].to_numpy(), return_inverse=True)
        item_ids, item_indices = np.unique(interactions["item_id"].to_numpy(), return_inverse=True)

        interaction_counts = (
            interactions["interaction_count"].to_numpy().astype(np.float32, copy=False)
        )

        user_items = csr_matrix(
            (interaction_counts, (user_indices, item_indices)),
            shape=(user_ids.size, item_ids.size),
            dtype=np.float32,
        )
        user_items.sort_indices()

        return (
            user_items,
            user_ids.astype(np.uint32, copy=False),
            item_ids.astype(np.uint32, copy=False),
        )
