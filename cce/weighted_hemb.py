import torch
import torch.nn as nn
from .hash import PolyHash
from .cce_robe import rolling_window


class WeightedHashEmbedding(nn.Module):
    def __init__( self, rows: int, dim: int, n_chunks: int, same=True):
        super().__init__()
        if same:
            rows *= 2
        self.same = same
        self.dim = dim
        self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows)
        self.table = nn.Parameter(torch.empty(rows, dim))
        # We can always discuss the best trade-off between these two.
        # For now we just set them to have the same size.
        self.hash1 = PolyHash(num_hashes=n_chunks, output_range=rows*dim)
        # TODO: Here's a crazy idea: What if we just use the same weight tensor
        # for the weights that we use for the rows? That is, we just reshape to
        # flat and index into the same memory...
        self.scale = nn.Parameter(torch.empty(1))
        #self.scale = nn.Parameter(torch.empty(n_chunks))
        #self.bias = nn.Parameter(torch.empty(n_chunks))
        #self.norm = nn.LayerNorm(n_chunks, elementwise_affine=False)
        if not same:
            self.weights = nn.Parameter(torch.empty(rows * dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table, -(self.dim**-0.5), self.dim**-0.5)
        with torch.no_grad():
            self.scale[:] = self.dim**.5
        if not self.same:
            nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        # table[hs].shape will be (batch_size, num_hashes, dim)
        #vecs = self.table[self.hash0(x)]
        vecs = self.table[self.hash0(x)]
        #vecs = vecs * self.weights(x)[:, :, None]
        #vecs = vecs * self.weights[self.hash1(x)]
        if self.same:
            weights = self.table.reshape(-1)[self.hash1(x)]
            #norm = torch.linalg.norm(weights, dim=-1)[:, None]
            weights = weights * self.scale
            #print(self.scale, self.bias)
            #weights = self.norm(weights)
            weights = weights[:, :, None]
        else:
            weights = self.weights[self.hash1(x)]
        #weights = 2 * self.hash1(x)[:, :, None] / self.hash1.range - 1
        vecs = vecs * weights
        return vecs.mean(dim=1)
