import torch
from torch import nn


class MultiHash(nn.Module):
    def __init__(self, num_hashes: int, output_bits: int):
        super().__init__()
        self.num_hashes = num_hashes
        self.output_bits = output_bits
        self.range = 2**output_bits
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
