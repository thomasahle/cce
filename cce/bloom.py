import torch
import torch.nn as nn
from .hash import PolyHash

# This is an unweighted version of Hash Embeddings.
# Often refereced as "Bloom Embeddings" following https://arxiv.org/abs/1706.03993
# See also https://thinc.ai/docs/api-layers#hashembed

class BloomEmbedding(nn.Module):
    def __init__(
        self,
        rows: int,
        dim: int,
        hash,
        signed: bool = False,
    ):
        super().__init__()
        self.hash = hash
        assert hash.range <= rows
        self.table = nn.Parameter(torch.empty(rows, dim))
        self.signed = signed
        if signed:
            self.sign_hash = PolyHash(hash.num_hashes, 2)
        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, x):
        # table[hs].shape will be (batch_size, num_hashes, dim)
        vecs = self.table[self.hash(x)]
        if self.signed:
            signs = self.sign_hash(x) * 2 - 1
            vecs = vecs * signs[:, :, None]
        return vecs.mean(dim=1)
