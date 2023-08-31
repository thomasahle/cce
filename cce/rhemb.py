import torch
import torch.nn as nn
from .hash import PolyHash
from .robe import get_slices


class RobeWeightedHashEmbedding(nn.Module):
    def __init__(self, size: int, dim: int, n_chunks: int, sparse=False):
        super().__init__()
        self.dim = dim
        self.n_chunks = n_chunks
        self.sparse = sparse
        self.hash0 = PolyHash(num_hashes=n_chunks, output_range=size//2)
        self.hash1 = PolyHash(num_hashes=n_chunks, output_range=size//2)
        self.table = nn.Parameter(torch.empty(size))
        self.table = nn.Parameter(torch.empty(size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table, -(self.dim**-0.5), self.dim**-0.5)

    def forward(self, x):
        # (batch_size, num_hashes, dim)
        vecs = get_slices(self.table, self.hash0(x), self.dim, self.sparse)
        # (batch_size, num_hashes, 1)
        weights = self.table[self.hash1(x)].unsqueeze(-1)
        scale = (self.n_chunks * self.dim) ** 0.5
        return (vecs * weights).mean(dim=1) * scale

