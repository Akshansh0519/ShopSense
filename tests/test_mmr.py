import numpy as np

from recommender.models.mmr import MMRReranker


def test_mmr_prefers_less_redundant_second_item():
    candidates = [(0, 1.0), (1, 0.99), (2, 0.90)]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )

    reranked = MMRReranker(lambda_param=0.5).rerank(candidates, embeddings, final_k=2)

    assert reranked[0][0] == 0
    assert reranked[1][0] == 2
