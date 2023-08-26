import torch
import torch.nn as nn
from cce import hash


class HashNetEmbedding(nn.Module):
    def __init__(
        self,
        size: int,
        hash: hash.MultiHash,
        sparse=False,
    ):
        super().__init__()
        self.hash = hash
        assert hash.range <= size
        self.sparse = sparse
        self.table = nn.Parameter(torch.empty(size))
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.hash.num_hashes
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, x):
        # table[hs].shape will be (batch_size, num_hashes)
        # return self.table[self.hash(x)]
        hs = self.hash(x)
        shape = hs.shape
        return torch.gather(
            self.table, 0, hs.flatten(), sparse_grad=self.sparse
        ).reshape(shape)
