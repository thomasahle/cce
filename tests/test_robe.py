import pytest
import torch
import torch.nn as nn
from itertools import product

import cce


@pytest.mark.parametrize(
    "chunk_size, num_chunks, table_size_log",
    product(range(1, 5), range(1, 5), [1, 2, 10]),
)
def test_robe(chunk_size, num_chunks, table_size_log):
    emb = cce.RobeEmbedding(
        2**table_size_log,
        chunk_size,
        cce.MultiHash(num_chunks, table_size_log),
    )
    n = 100
    hash_values = torch.randint(2**62, size=(n,))
    output = emb(hash_values)
    assert output.shape == (n, chunk_size * num_chunks)


def test_empty_tensor():
    emb = cce.RobeEmbedding(16, 3, cce.MultiHash(2, 4))
    hash_values = torch.tensor([], dtype=torch.long)
    output = emb(hash_values)
    assert output.shape == (0, 3 * 2)


def test_stability():
    emb = cce.RobeEmbedding(16, 3, cce.MultiHash(2, 4))
    hash_values = torch.randint(2**62, size=(100,))
    output1 = emb(hash_values)
    output2 = emb(hash_values)
    assert torch.equal(output1, output2)


class IdentityHash:
    def __init__(self, num_hashes, table_size_log):
        self.num_hashes = num_hashes

    def __call__(self, input_tensor):
        # Repeat the identity hash values for the desired number of chunks.
        return input_tensor.unsqueeze(-1).repeat(1, self.num_hashes)


def test_identity_hash_with_arange_table():
    size = 8
    chunk_size = 3
    num_chunks = 2

    # Using the modified RobeEmbedding with IdentityHash
    emb = cce.RobeEmbedding(size, chunk_size, IdentityHash(num_chunks, 4))
    emb.table = nn.Parameter(torch.arange(size, dtype=torch.float))

    # For example, choose hash values of [1, 2, 3]
    hash_values = torch.tensor([1, 3, 6])
    output = emb(hash_values)

    # Expected output is slices from the table:
    # hash_values of 1 -> table[1], table[1+1], table[1+2]
    # hash_values of 2 -> table[2], table[2+1], table[2+2]
    # and so on for each hash value, and then repeated for num_chunks times.
    expected_output = torch.tensor([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5], [6, 7, 0, 6, 7, 0]])

    assert torch.equal(output, expected_output)
