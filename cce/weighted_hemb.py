import torch
import torch.nn as nn
from .hash import PolyHash
from .cce_robe import rolling_window


class WeightedHashEmbedding(nn.Module):
    def __init__( self, rows: int, dim: int, n_chunks: int, roll=False):
        super().__init__()
        self.dim = dim
        self.roll = roll
        if roll:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows*dim)
            self.table = nn.Parameter(torch.empty(rows * dim))
        else:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows)
            self.table = nn.Parameter(torch.empty(rows, dim))
        # We can always discuss the best trade-off between these two.
        # For now we just set them to have the same size.
        self.hash1 = PolyHash(num_hashes=n_chunks, output_range=rows*dim)
        self.weights = nn.Parameter(torch.empty(rows * dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table, -(self.dim**-0.5), self.dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        # table[hs].shape will be (batch_size, num_hashes, dim)
        #vecs = self.table[self.hash0(x)]
        if self.roll:
            vecs = rolling_window(self.table, self.dim)[self.hash0(x)]
        else:
            vecs = self.table[self.hash0(x)]
        #vecs = vecs * self.weights(x)[:, :, None]
        #vecs = vecs * self.weights[self.hash1(x)]
        weights = self.weights[self.hash1(x)]
        #weights = 2 * self.hash1(x)[:, :, None] / self.hash1.range - 1
        vecs = vecs * weights
        return vecs.mean(dim=1)
