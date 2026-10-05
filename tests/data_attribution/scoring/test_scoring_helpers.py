import torch
import logging
from data_attribution.attribution.run import _score_with_gradient_sets


def test_score_with_gradient_sets_full_matrix():
    # Simulate a scenario with N_train=100, N_query=50, Emb_dim=10
    n_train = 100
    n_query = 50
    dim = 10

    train_grads = {"layer1": torch.randn(n_train, dim)}
    query_grads = {"layer1": torch.randn(n_query, dim)}

    logger = logging.getLogger("test")

    # This calls the function which currently computes the full matrix
    scores = _score_with_gradient_sets(train_grads, query_grads, k=5, logger=logger)

    # Verify the shape is (N_train, N_query)
    assert scores.shape == (n_train, n_query)

    # If we had huge dimensions, this would OOM.
    # The fix should avoid returning this full matrix if integrated with top-k extraction.
