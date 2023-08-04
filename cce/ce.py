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
        nn.init.uniform_(self.table)

    def forward(self, x):
        rows, n_chunks, chunk_size = self.table.shape
        dim = n_chunks * chunk_size
        return self.table[self.hash(x), range(n_chunks)].reshape(-1, dim)
