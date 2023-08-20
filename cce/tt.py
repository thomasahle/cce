# [ ] - [ ] - [ ]
#  |     |     |
# 
#  The simple case is just low-rank matrix
# [ ] - [ ]
#  |     |
# 
#  (UV)_{ij} = e_i U V e_j = <U_i, V_j>
# 
#  So it's  more like I have a bunch of stacks of matrices, and I pick one from each.
#  And from the ends I just pick a vector.

# Seems there are two ways I could do this. There is
# [ ] - [ ] - [ ] - out_dim
#  |     |     |
# hra   hra   hra

# Or there is
# Seems there are two ways I could do this. There is

#     out_dim
#        |
#  /-----+-----\
#  |     |     |
# [ ] - [ ] - [ ]
#  |     |     |
# hra   hra   hra

# The first will be cheaper.
# But the second costs a factor out_dim more.

# I guess there's also something like:
#
# out   out   out
#  |     |     |
# [ ] - [ ] - [ ]
#  |     |     |
# hra   hra   hra
#
# which will only incur a factor d^(1/chunks). So it's a compromise.
# I think this is actually what the TT paper does. Since they mention
# that each tensor is 4d.
# Also that they reduce from O(MN) to O(d R^2 max(m,n)^2)
# Actually, shouldn't it be d R^2 m n?
# (Here d is the number of chunks)

import torch
import torch.nn as nn
from cce import hash

# To save the most, we should make c ~ output_bits.
# But maybe not a good idea in for quality.
# log(c) + log(hrange)/c
# 1/c = log(hrange)/c^2
# 1 = log(hrange)/c
# c = log(hrange)


class TensorTrainEmbedding(nn.Module):
    # total params: chunks * hrange^(1/chunks) * dim * rank^2
    def __init__(
        self,
        rank: int,
        dim: int,
        hash: hash.MultiHash,
    ):
        super().__init__()
        self.hash = hash
        n_chunks, = hash.hash_coeffs.shape
        assert n_chunks >= 2
        assert hash.output_bits >= 1
        hrange = 2 ** hash.output_bits
        print(f'{hrange=}')
        self.start_core = nn.Parameter(torch.empty(hrange, dim, rank))
        self.end_core = nn.Parameter(torch.empty(hrange, dim, rank))
        self.cores = nn.Parameter(torch.empty(n_chunks - 2, hrange, dim, rank, rank))
        self.reset_parameters()

    def reset_parameters(self):
        _, _, dim, rank, _ = self.cores.shape
        nn.init.uniform_(self.cores, -rank**-0.5, rank**-0.5)
        nn.init.uniform_(self.end_core, -(rank**-0.5), rank**-0.5)
        nn.init.uniform_(self.start_core, -(dim**-0.5), dim**-0.5)

    def forward(self, x):
        hs = self.hash(x)
        #print(x, hs)
        v = self.end_core[hs[:, -1]] # (batch, dim, rank)
        for i in range(hs.shape[1]-2):
            hi = hs[:, i+1]
            v = torch.einsum('bdrs,bdr->bds', self.cores[i, hi], v)
        return torch.einsum('bdr,bdr->bd', self.start_core[hs[:, 0]], v)
