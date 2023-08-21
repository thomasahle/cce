import math
import torch
from torch import nn


class MultiHash(nn.Module):
    def __init__(self, num_hashes: int, output_range: int):
        super().__init__()
        self.num_hashes = num_hashes
        self.output_bits = int(math.ceil(math.log2(output_range)))
        self.range = output_range
        self.hash_coeffs = nn.Parameter(
            torch.randint(1 << 62, size=(num_hashes,)), requires_grad=False
        )

    def forward(self, input_tensor):
        """[x1, x2, x3, ...] -> [[h1(x1), h2(x1)], [h2(x1), h2(x2)], ...]"""
        return (
            ((input_tensor[..., None] * self.hash_coeffs) & (2**62 - 1))
            >> 62 - self.output_bits
        ) % self.range


class QRHash(nn.Module):
    # Not a real hash, instead splits into words using modulo like QR

    def __init__(self, num_hashes: int, output_range: int):
        super().__init__()
        self.num_hashes = num_hashes
        self.range = output_range

    def forward(self, input_tensor):
        """[x1, x2, x3, ...] -> [[h1(x1), h2(x1)], [h2(x1), h2(x2)], ...]"""
        res = []
        for _ in range(self.num_hashes):
            res.append(input_tensor % self.range)
            input_tensor = input_tensor // self.range
        return torch.stack(res, dim=-1)


class PolyHash(nn.Module):
    # May have a slightly better distribution than MultiHash above, but only
    # supports inputs up to 2^31-1, where MultiHash goes up to 2^63.

    def __init__(self, num_hashes: int, output_range: int):
        super().__init__()
        self.num_hashes = num_hashes
        self.range = output_range
        assert output_range <= 2**31 - 1, "Must be less than the Mersenne 2^31-1"
        self.cs = nn.Parameter(
            torch.randint(1, 1 << 31 - 1, size=(num_hashes,)), requires_grad=False
        )

    def forward(self, input_tensor):
        """[x1, x2, x3, ...] -> [[h1(x1), h2(x1)], [h2(x1), h2(x2)], ...]"""
        x = input_tensor[..., None] * self.cs
        # Could speed this up a bit using fancy Mersenne output tricks
        return x % (2**31 - 1) % self.range

