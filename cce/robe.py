import torch
import torch.nn as nn
from cce import hash


class RobeEmbedding(nn.Module):
    def __init__(
        self,
        size: int,
        chunk_size: int,
        hash: hash.MultiHash,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hash = hash
        self.table = nn.Parameter(torch.empty(size))
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.chunk_size * self.hash.hash_coeffs.shape[0]
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape
        hash_values = self.hash(input_tensor)  # (batch_size, num_hashes)
        slices = self.table[
            hash_values[..., None]
            + torch.arange(self.chunk_size, device=hash_values.device) % len(self.table)
        ]  # (batch_size, num_hashes, chunk_size)
        return slices.reshape(*batch_size, -1)
