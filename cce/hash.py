import torch
from torch import nn


def multiply_top_62(a, b):
    """
    Multiplies two 62 bit intergers and returns the top 62 bits.
    Works with torch.long, so converstion to python int is not needed.
    """

    # Unfortunately torch doesn't support unsigned ints
    assert a.dtype == b.dtype == torch.int64

    # We assume the input is 62 bits, so we don't have to worry about carries.
    assert torch.all(a < 2**62) and torch.all(b < 2**62)

    # Split the inputs into two 31-bit integers
    high_a, low_a = a >> 31, a & 0x7FFFFFFF
    high_b, low_b = b >> 31, b & 0x7FFFFFFF

    # Perform the multiplication in parts, taking care to move the carry
    # up each step
    x = (low_a * low_b) >> 31
    x = (x + high_a * low_b + high_b * low_a) >> 31
    x = x + high_a * high_b
    return x


class MultiHash(nn.Module):
    def __init__(self, num_hashes: int, output_bits: int):
        super().__init__()
        self.output_bits = output_bits
        self.hash_coeffs = nn.Parameter(
            torch.randint(1 << 62, size=(num_hashes,)), requires_grad=False
        )

    def forward(self, input_tensor):
        """[x1, x2, x3, ...] -> [[h1(x1), h2(x1)], [h2(x1), h2(x2)], ...]"""
        return (
            (input_tensor[..., None] * self.hash_coeffs) & (2**62 - 1)
        ) >> 62 - self.output_bits


class SingleHash(MultiHash):
    def __init__(self, output_bits: int):
        super().__init__(1, output_bits)

    def forward(self, input_tensor):
        """[x1, x2, x3, ...] -> [h(x1), h(x2), h(x3), ...]"""
        return super().forward(input_tensor).squeeze(-1)
