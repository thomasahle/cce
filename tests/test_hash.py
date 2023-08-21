import pytest
import torch
from itertools import product
import math

from cce import hash


@pytest.mark.parametrize("num_hashes, output_bits", product([100], range(5, 7)))
def test_multi(num_hashes, output_bits):
    h = hash.MultiHash(num_hashes, 2**output_bits)
    n = 10000
    x = torch.randint(2**62, size=(n,))
    hs = h(x)
    assert hs.shape == (n, num_hashes)
    assert torch.all(hs < 2**output_bits)

    for i in range(num_hashes):
        # Each bin is approximately ~ Poisson(mu) where mu = n / bins.
        # So the variance is also mu and we expect std/mu**.5 to be small.
        std = torch.bincount(hs[:, i], minlength=2**output_bits).to(float).std()
        assert std / (n / 2**output_bits) ** 0.5 < 2
