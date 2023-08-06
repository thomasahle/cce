import torch
from torch import nn


class MultiHash(nn.Module):
    def __init__(self, num_hashes: int, output_bits: int):
        super().__init__()
        self.num_hashes = num_hashes
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
