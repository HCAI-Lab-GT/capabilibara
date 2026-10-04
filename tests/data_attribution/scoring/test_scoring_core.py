import torch

from data_attribution.scoring.core import (
    batched_topk,
    score_batch,
    select_gradient_keys,
)


def test_select_gradient_keys_preferred() -> None:
    training = {"g": torch.zeros((1, 2))}
    query = {"g": torch.zeros((1, 2))}
    assert select_gradient_keys(training, query, "g") == ["g"]


def test_select_gradient_keys_overlap() -> None:
    training = {"a": torch.zeros((1, 2)), "b": torch.zeros((1, 2))}
    query = {"b": torch.zeros((1, 2))}
    assert select_gradient_keys(training, query, None) == ["b"]


def test_score_batch_topk() -> None:
    training = {"g": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    query = {"g": torch.tensor([[1.0, 0.0], [0.2, 0.8]])}
    scores, indices = score_batch(training, query, ["g"], k=1)
    assert indices[:, 0].tolist() == [0]
    assert indices[:, 1].tolist() == [1]
    assert scores.shape == (1, 2)


def test_batched_topk_batches() -> None:
    training = {"g": torch.eye(2)}
    query = {"g": torch.tensor([[1.0, 0.0]] * 5)}
    starts = [
        start
        for start, _, _ in batched_topk(training, query, keys=["g"], k=1, batch_size=2)
    ]
    assert starts == [0, 2, 4]
