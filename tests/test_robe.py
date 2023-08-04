import pytest
import torch
from itertools import product

import cce


@pytest.mark.parametrize(
    "chunk_size, num_chunks, table_size",
    product(range(1, 5), range(1, 5), [1, 2, 2**10])
)
def test_robe(chunk_size, num_chunks, table_size):
    emb = cce.RobeEmbedding(
        table_size,
        chunk_size,
        cce.MultiHash(num_chunks, table_size),
    )
    n = 100
    hash_values = torch.randint(2**62, size=(n,))
    output = emb(hash_values)
    assert output.shape == (n, chunk_size * num_chunks)
