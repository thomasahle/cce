import torch
import torch.nn as nn
from cce import hash


class HashEmbedding(nn.Module):
    def __init__(
        self,
        rows: int,
        dim: int,
        hash: hash.MultiHash,
    ):
        super().__init__()
        self.hash = hash
        assert hash.range <= rows
        self.table = nn.Parameter(torch.empty(rows, dim))
        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, x):
        # Shape will be (batch_size, num_hashes, dim)
        return self.table[self.hash(x)].mean(dim=1)
