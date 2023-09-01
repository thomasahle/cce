import torch
import torch.nn as nn
from .hash import PolyHash
from .robe import get_slices

# Based on Dan Tito Svenstrup, Jonas Hansen, Ole Winther
# https://papers.nips.cc/paper_files/paper/2017/hash/f0f6ba4b5e0000340312d33c212c3ae8-Abstract.html

class HashEmbedding(nn.Module):
    def __init__(
        self,
        num_params: int,
        dim: int,
        n_hash: int,  # Paper says "we typically use k = 2"
    ):
        super().__init__()
        # We pick B and K so B*dim = K*n_hash, while respecting num_params.
        # This fits roughly with the paper specification "K > 10 B".
        B = max(num_params // (2 * dim), 1)
        K = max(num_params // (2 * n_hash), 1)
        self.table = nn.Parameter(torch.empty(B, dim))
        #self.weights = nn.Parameter(torch.empty(K, n_hash))
        self.weights = nn.Parameter(torch.empty(K * n_hash))

        print(self.table.numel() + self.weights.numel(), num_params)
        assert self.table.numel() + self.weights.numel() <= num_params

        self.hash0 = PolyHash(num_hashes=n_hash, output_range=B)
        #self.hash1 = hash.PolyHash(num_hashes=1, output_range=K)
        self.hash1 = PolyHash(num_hashes=n_hash, output_range=K * n_hash)
        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        vecs = self.table[self.hash0(x)]  # (batch_size, num_hashes, dim)
        #weights = self.weights[self.hash1(x)]  # (batch_size, 1, num_hashes)
        weights = self.weights[self.hash1(x)].unsqueeze(1)  # (batch_size, 1, num_hashes)
        #weights = torch.softmax(weights, dim=1)
        return (weights @ vecs).squeeze(1)  # (batch_size, dim)


class HashEmbedding2(nn.Module):
    """ Using the "optional" strategy of concatenating the weights with the vector. """
    def __init__(
        self,
        num_params: int,
        dim: int,
        n_hash: int,
    ):
        super().__init__()
        self.dim = dim
        # We pick B and K so B*dim = K*n_hash, while respecting num_params.
        # This fits roughly with the paper specification "K > 10 B".
        B = max(num_params // (2 * (dim - n_hash)), 1)
        K = max(num_params // (2 * n_hash), 1)
        self.table = nn.Parameter(torch.empty(B, dim - n_hash))
        self.weights = nn.Parameter(torch.empty(K, n_hash))

        assert self.table.numel() + self.weights.numel() <= num_params

        self.hash0 = PolyHash(num_hashes=n_hash, output_range=B)
        self.hash1 = PolyHash(num_hashes=1, output_range=K)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table, -(self.dim**-0.5), self.dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        vecs = self.table[self.hash0(x)]  # (batch_size, num_hashes, dim)
        weights = self.weights[self.hash1(x)]  # (batch_size, 1, num_hashes)
        res = torch.cat([weights @ vecs, weights / self.dim**.5], dim=2)  # (bs, 1, dim + num_hashes)
        return res.squeeze(1)  # (batch_size, dim)


class WeightedHashEmbedding(nn.Module):
    def __init__(self, rows: int, dim: int, n_chunks: int, sparse=False, share_table=True):
        super().__init__()
        self.dim = dim
        self.n_chunks = n_chunks
        self.sparse = sparse

        self.share_table = share_table
        if share_table:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows)
            self.hash1 = PolyHash(num_hashes=n_chunks, output_range=rows * dim)
            self.table0 = nn.Parameter(torch.empty(rows, dim))
            self.table1 = self.table0
        else:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows // 2)
            self.hash1 = PolyHash(num_hashes=n_chunks, output_range=rows * dim // 2)
            self.table0 = nn.Parameter(torch.empty(rows // 2, dim))
            self.table1 = nn.Parameter(torch.empty(rows * dim // 2))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table0, -(self.dim**-0.5), self.dim**-0.5)
        if not self.share_table:
            nn.init.uniform_(self.table1, -1, 1)

    def forward(self, x):
        vecs = self.table0[self.hash0(x)]

        weights = self.table1.view(-1)[self.hash1(x)].unsqueeze(-1)

        if self.share_table:
            scale = (self.n_chunks * self.dim) ** 0.5
        else:
            scale = (self.n_chunks) ** 0.5

        return (vecs * weights).mean(dim=1) * scale


class RobeWeightedHashEmbedding(nn.Module):
    def __init__(self, size: int, dim: int, n_chunks: int, sparse=False, share_table=True):
        super().__init__()
        self.dim = dim
        self.n_chunks = n_chunks
        self.share_table = share_table
        if share_table:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=size)
            self.hash1 = PolyHash(num_hashes=n_chunks, output_range=size)
            self.table0 = nn.Parameter(torch.empty(size))
            self.table1 = self.table0
        else:
            self.hash0 = PolyHash(num_hashes=n_chunks, output_range=size//2)
            self.hash1 = PolyHash(num_hashes=n_chunks, output_range=size//2)
            self.table0 = nn.Parameter(torch.empty(size//2))
            self.table1 = nn.Parameter(torch.empty(size//2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table0, -(self.dim**-0.5), self.dim**-0.5)
        if not self.share_table:
            nn.init.uniform_(self.table1, -1, 1)

    def forward(self, x):
        # (batch_size, num_hashes, dim)
        vecs = get_slices(self.table0, self.hash0(x), self.dim)
        # (batch_size, num_hashes, 1)
        weights = self.table1[self.hash1(x)].unsqueeze(-1)
        if self.share_table:
            scale = (self.n_chunks * self.dim) ** 0.5
        else:
            scale = (self.n_chunks) ** 0.5
        return (vecs * weights).mean(dim=1) * scale

