import torch
from torch import nn


def multiply_top_64(a, b):
    # Split the inputs into two 32-bit integers
    high_a, low_a = a >> 32, a & 0xFFFFFFFF
    high_b, low_b = b >> 32, b & 0xFFFFFFFF

    # Perform the multiplication in parts
    high_product = high_a * high_b
    cross_product = high_a * low_b + high_b * low_a
    # We only need to keep the high 32 bits of the cross product
    cross_product = cross_product >> 32

    # Add up the components to get the top 64 bits of the multiplication
    top_64 = high_product + cross_product

    return top_64


class MultiHash(nn.Module):
    def __init__(self, num_hashes: int, output_bits: int):
        super().__init__()
        self.output_bits = output_bits
        self.hash_coeffs = nn.Parameter(
            torch.randint((1 << 63) - 1, size=(num_hashes,)),
            requires_grad=False
        )

    def forward(self, input_tensor):
        """ [x1, x2, x3, ...] -> [[h1(x1), h2(x1)], [h2(x1), h2(x2)], ...] """
        return (
            multiply_top_64(input_tensor[..., None], self.hash_coeffs)
            >> 63 - self.output_bits
        )
