import torch
import torch.nn as nn
from cce import hash

# Based on Dan Tito Svenstrup, Jonas Hansen, Ole Winther
# https://papers.nips.cc/paper_files/paper/2017/hash/f0f6ba4b5e0000340312d33c212c3ae8-Abstract.html

class HashEmbedding(nn.Module):
    def __init__(
        self,
        num_params: int,
        dim: int,
        n_hash: int = 2,  # Paper says "we typically use k = 2"
    ):
        super().__init__()
        # We pick B and K so B*dim = K*n_hash, while respecting num_params.
        # This fits roughly with the paper specification "K > 10 B".
        B = max(num_params // (2 * n_hash), 1)
        K = max(num_params // (2 * dim), 1)
        self.table = nn.Parameter(torch.empty(B, dim))
        #self.weights = nn.Parameter(torch.empty(K, n_hash))
        self.weights = nn.Parameter(torch.empty(K * n_hash))

        self.hash0 = hash.PolyHash(num_hashes=n_hash, output_range=B)
        #self.hash1 = hash.PolyHash(num_hashes=1, output_range=K)
        self.hash1 = hash.PolyHash(num_hashes=n_hash, output_range=K * n_hash)
        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        vecs = self.table[self.hash0(x)]  # (batch_size, num_hashes, dim)
        #weights = self.weights[self.hash1(x)]  # (batch_size, 1, num_hashes)
        weights = self.weights[self.hash1(x)].unsqueeze(1)  # (batch_size, 1, num_hashes)
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
        # We pick B and K so B*dim = K*n_hash, while respecting num_params.
        # This fits roughly with the paper specification "K > 10 B".
        B = max(num_params // (2 * n_hash), 1)
        K = max(num_params // (2 * (dim - n_hash)), 1)
        self.table = nn.Parameter(torch.empty(B, dim - n_hash))
        self.weights = nn.Parameter(torch.empty(K, n_hash))

        self.hash0 = hash.PolyHash(num_hashes=n_hash, output_range=B)
        self.hash1 = hash.PolyHash(num_hashes=1, output_range=K)
        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        vecs = self.table[self.hash0(x)]  # (batch_size, num_hashes, dim)
        weights = self.weights[self.hash1(x)]  # (batch_size, 1, num_hashes)
        res = torch.cat([weights @ vecs, weights], dim=2)  # (bs, 1, dim + num_hashes)
        return res.squeeze(1)  # (batch_size, dim)
