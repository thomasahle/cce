import pytest
import torch
import torch.nn as nn
from itertools import product

import cce


@pytest.mark.parametrize(
    "vocab, num_params, dimension, method, n_chunks, sparse",
    [(512, 1024, 16, method, 4, False) for method in cce.methods],
)
def test_make_embedding(vocab, num_params, dimension, method, n_chunks, sparse):
    bs = 100
    emb = cce.make_embedding(vocab, num_params, dimension, method, n_chunks, sparse)
    input = torch.randint(vocab, size=(bs,))
    assert emb(input).shape == (bs, dimension)

    if hasattr(emb, "cluster"):
        emb.cluster()
        assert emb(input).shape == (bs, dimension)
