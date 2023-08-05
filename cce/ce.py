import torch
import torch.nn as nn
from cce import hash

class CompositionalEmbedding(nn.Module):
    def __init__(
        self,
        rows: int,
        chunk_size: int,
        hash: hash.MultiHash,
    ):
        super().__init__()
        self.hash = hash
        n_chunks, = hash.hash_coeffs.shape
        self.table = nn.Parameter(torch.empty(rows, n_chunks, chunk_size))
        self.reset_parameters()

    def reset_parameters(self):
        rows, n_chunks, chunk_size = self.table.shape
        dim = chunk_size * n_chunks
        nn.init.uniform_(self.table, -dim**-.5, dim**-.5)

    def forward(self, x):
        rows, n_chunks, chunk_size = self.table.shape
        return self.table[self.hash(x), range(n_chunks)].flatten(1, 2)
