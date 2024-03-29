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
import math
from cce import hash


class TensorTrainEmbedding(nn.Module):
    # total params: chunks * hrange^(1/chunks) * dim * rank^2
    def __init__(
        self,
        rank: int,
        dim: int,
        hash: hash.MultiHash,
        split_dim=True,
    ):
        super().__init__()
        # For TT we use a QR "Hash" which doesn't have collisions.
        # This hash makes sense for TT-Rec more than the other methods, since
        # TT-Rec tends to have much smaller ranges than the other methods.
        self.hash = hash
        self.n_chunks = n_chunks = hash.num_hashes
        assert n_chunks >= 2
        self.rank = rank

        hrange = hash.range
        assert hrange >= 1

        # The reference implementation of TT doesn't just split the input space,
        # but also the output space. We do the same if split_dim=True. However,
        # we may sometimes overshoot, if dim^(1/n_chunks) is not an interger.
        # In this case we just crop the output dimension at the end, in forward().
        self.dim = dim
        self.split_dim = split_dim
        if split_dim:
            dim = int(math.ceil(dim ** (1 / n_chunks)))

        self.start_core = nn.Parameter(torch.empty(hrange, dim, rank))
        self.end_core = nn.Parameter(torch.empty(hrange, dim, rank))
        self.cores = nn.Parameter(torch.empty(n_chunks - 2, hrange, dim, rank, rank))
        self.reset_parameters()

    def reset_parameters(self):
        # To get unit-norm output vectors, as we generally want for DLRM type
        # systems, we need to scale each core by 1/sqrt(rank * dim), except for
        # start, which is only scaled by 1/sqrt(dim).
        _, dim, rank = self.start_core.shape
        if self.split_dim:
            scale = (dim * rank) ** -0.5
            nn.init.uniform_(self.cores, -scale, scale)
            nn.init.uniform_(self.end_core, -scale, scale)
            nn.init.uniform_(self.start_core, -(dim**-0.5), dim**-0.5)
        else:
            scale = rank**-0.5
            nn.init.uniform_(self.cores, -scale, scale)
            nn.init.uniform_(self.end_core, -scale, scale)
            nn.init.uniform_(self.start_core, -(dim**-0.5), dim**-0.5)

    def size(self):
        return self.start_core.numel() + self.end_core.numel() + self.cores.numel()

    def forward(self, x):
        hs = self.hash(x)
        v = self.end_core[hs[:, -1]]  # (batch, dim, rank)
        if not self.split_dim:
            for i in range(hs.shape[1] - 2):
                hi = hs[:, i + 1]
                v = torch.einsum("bdrs,bdr->bds", self.cores[i, hi], v)
            v = torch.einsum("bdr,bdr->bd", self.start_core[hs[:, 0]], v)
        else:
            for i in range(hs.shape[1] - 2):
                core = self.cores[i, hs[:, i + 1]]
                v = torch.einsum("bdsr,ber->bdes", core, v)
                v = v.flatten(1, 2)
            v = torch.einsum("bdr,ber->bde", self.start_core[hs[:, 0]], v)
            v = v.flatten(1, 2)
            v = v[:, : self.dim]
        return v
