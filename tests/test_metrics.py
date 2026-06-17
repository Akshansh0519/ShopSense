import numpy as np

from recommender.evaluation.metrics import (
    diversity_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ranking_metrics_on_known_example():
    recommended = [1, 2, 3, 4]
    actual = {2, 4}

    assert recall_at_k(recommended, actual, k=4) == 1.0
    assert precision_at_k(recommended, actual, k=4) == 0.5
    assert round(ndcg_at_k(recommended, actual, k=4), 4) == 0.6509
    assert map_at_k(recommended, actual, k=4) == (1 / 2 + 2 / 4) / 2


def test_diversity_rewards_different_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    redundant = diversity_at_k([0, 1], embeddings, k=2)
    diverse = diversity_at_k([0, 2], embeddings, k=2)

    assert diverse > redundant
