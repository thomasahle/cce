import pytest
import torch
from itertools import product

from cce import hash


def test_mult_62():
    x = torch.randint(2**62, size=(100,))
    y = torch.randint(2**62, size=(100,))
    truth = [(int(xi) * int(yi)) >> 62 for xi, yi in zip(x, y)]
    truth = torch.tensor(truth, dtype=torch.long)
    result = hash.multiply_top_62(x, y)
    torch.testing.assert_allclose(truth, result)


@pytest.mark.parametrize(
    "num_hashes, output_bits",
    product([100], range(5, 7))
)
def test_multi(num_hashes, output_bits):
    h = hash.MultiHash(num_hashes, output_bits)
    n = 10000
    x = torch.randint(2**62, size=(n,))
    hs = h(x)
    assert hs.shape == (n, num_hashes)
    assert torch.all(hs < 2**output_bits)

    for i in range(num_hashes):
        # Each bin is approximately ~ Poisson(mu) where mu = n / bins.
        # So the variance is also mu and we expect std/mu**.5 to be small.
        std = torch.bincount(hs[:, i], minlength=2**output_bits).to(float).std()
        assert std / (n / 2**output_bits) ** .5 < 2



