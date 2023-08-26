import torch
import torch.nn as nn


class CompositionalEmbedding(nn.Module):
    def __init__(
        self,
        rows: int,
        chunk_size: int,
        hash,
        sparse=False,
    ):
        super().__init__()
        self.hash = hash
        self.sparse = sparse
        n_chunks = hash.num_hashes
        self.table = nn.Parameter(torch.empty(rows, n_chunks, chunk_size))
        self.reset_parameters()

    def reset_parameters(self):
        rows, n_chunks, chunk_size = self.table.shape
        dim = chunk_size * n_chunks
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, x):
        rows, n_chunks, chunk_size = self.table.shape
        hs = self.hash(x)[:, :, None]
        hs = hs.expand(-1, n_chunks, chunk_size)
        gathered = torch.gather(self.table, 0, hs, sparse_grad=self.sparse)
        return gathered.flatten(1, 2)
