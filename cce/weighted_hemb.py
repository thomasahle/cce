import torch
import torch.nn as nn
from .hash import PolyHash
from .cce_robe import rolling_window
from torch.autograd import Function


class SparseGradientFunction(Function):
    @staticmethod
    def forward(ctx, table, hash0_x, hash1_x, dim, n_chunks):
        vecs = table[hash0_x]
        weights = table.flatten()[hash1_x].unsqueeze(-1)
        scale = (n_chunks * dim) ** 0.5
        output = (vecs * weights).mean(dim=1) * scale
        ctx.save_for_backward(hash0_x, hash1_x, vecs, weights, table.size())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hash0_x, hash1_x, vecs, weights, table_size = ctx.saved_tensors

        grad_vecs = (grad_output.unsqueeze(1) * weights).mean(dim=1)

        # Create the indices for the sparse gradient based on hash0_x
        indices = hash0_x.t().contiguous().view(2, -1)
        values = grad_vecs.contiguous().view(-1)

        # Create a sparse tensor for grad_vecs
        sparse_grad_vecs = torch.sparse_coo_tensor(
            indices, values, table_size, device=table.device
        )

        # Handle the gradient for weights
        grad_weights = (vecs * grad_output.unsqueeze(1)).mean(dim=1).flatten()
        grad_table_flat_indices = hash1_x.view(1, -1)
        grad_table_flat_values = grad_weights.view(-1)

        # Create a sparse tensor for grad_weights
        sparse_grad_weights = torch.sparse_coo_tensor(
            grad_table_flat_indices,
            grad_table_flat_values,
            table.numel(),
            device=table.device,
        )

        # Combine the two sparse gradients
        combined_sparse_grad = (
            sparse_grad_vecs
            + sparse_grad_weights.coalesce().resize_as_(sparse_grad_vecs)
        ).coalesce()

        return combined_sparse_grad.to_dense(), None, None, None, None


class WeightedHashEmbedding(nn.Module):
    def __init__(self, rows: int, dim: int, n_chunks: int, sparse=False):
        super().__init__()
        self.dim = dim
        self.n_chunks = n_chunks
        self.sparse = sparse
        #rows //= 2
        self.hash0 = PolyHash(num_hashes=n_chunks, output_range=rows)
        self.hash1 = PolyHash(num_hashes=n_chunks, output_range=rows * dim)
        self.table = nn.Parameter(torch.empty(rows, dim))
        #self.table1 = nn.Parameter(torch.empty(rows * dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.table, -(self.dim**-0.5), self.dim**-0.5)
        #nn.init.uniform_(self.table1, -1, 1)

    def forward(self, x):
        vecs = self.table[self.hash0(x)]
        weights = self.table.flatten()[self.hash1(x)].unsqueeze(-1)
        #weights = self.table1[self.hash1(x)].unsqueeze(-1)
        scale = (self.n_chunks * self.dim) ** 0.5
        #scale = (self.n_chunks) ** 0.5
        return (vecs * weights).mean(dim=1) * scale

    def forward_sparse(self, x):
        hs = self.hash0(x).view(-1, 1).expand(-1, self.dim)
        vecs = torch.gather(self.table, 0, hs, sparse_grad=self.sparse)
        vecs = vecs.view(x.shape + (self.n_chunks, self.dim))
        hs = self.hash1(x)
        shape = hs.shape
        # TODO: We need table.flatten(), but this creates a reshape on the gradient,
        # and currently pytorch doesn't support reshape on sparse tensors.
        weights = torch.gather(
            self.table.flatten(), 0, hs.flatten(), sparse_grad=self.sparse
        )
        weights = weights.view(shape).unsqueeze(-1)
        # Scale to unit norm
        scale = (self.hash1.num_hashes * self.dim) ** 0.5
        return (vecs * weights).mean(dim=1) * scale

    def forward_sparse2(self, x):
        return SparseGradientFunction.apply(
            self.table, self.hash0(x), self.hash1(x), self.dim, self.n_chunks
        )
