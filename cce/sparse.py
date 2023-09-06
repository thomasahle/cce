import torch
import torch.nn as nn
import os
import time
import numpy as np
from torch.autograd import Function


def omp(X, D, s):
    """
    X: data matrix of size (n_samples, n_features)
    D: dictionary of size (n_atoms, n_features)
    s: desired sparsity level
    Returns the sparse code matrix S of size (n_samples, n_atoms)
    """
    n_samples, n_features = X.shape
    n_atoms, _ = D.shape

    # Normalize rows
    norms = torch.linalg.norm(D, axis=1) + 1e-6
    D2 = D / norms[:, None]

    residuals = X.clone()
    ids = torch.zeros((n_samples, s), dtype=int)

    for step in range(s):
        corrs = torch.abs(residuals @ D2.T)
        corrs.scatter_(1, ids[:, :step], -1)  # Mask ids we've already picked
        ids[:, step] = torch.argmax(corrs, axis=1)
        subDs = D2[ids[:, : step + 1]].mT
        m = torch.linalg.lstsq(subDs, X, rcond=None).solution
        residuals = X - (subDs @ m.unsqueeze(2)).squeeze(2)

    m /= norms[ids]
    return ids, m


def k_svd(X, M, s, n_iter, max_time=None):
    assert n_iter >= 1
    n, d = X.shape
    k, _ = M.shape

    start = time.time()
    for iter in range(n_iter):
        # Sparse Coding
        ids, m = omp(X, M, s)

        # Dictionary Update, one row at a time
        for j in range(k):
            # Find which samples are currently assumed to be using the kth atom
            mask = (ids == j)
            I = torch.any(mask, dim=1)

            if not torch.any(I):
                continue
            # Compute current residuals
            E = X[I] - (m[I].unsqueeze(1) @ M[ids[I]]).squeeze(1)  # E = X[I] - S[I] @ M
            # Add the current atom back in. This is like assuming it was
            # currently not used by any of the samples.
            E += torch.outer(m[mask], M[j])  # E += torch.outer(S[I, j], M[j])

            # Use svd_lowrank from torch to save time and only compute
            # the first singular vector
            U, Sigma, V = torch.svd_lowrank(E, q=1)
            # U, Sigma, V = torch.linalg.svd(E)
            M[j, :] = V[0, :]
            # We also update S, which is how k-svd can have an advantage over MOD
            # S[I, j] = Sigma[0] * U[:, 0]
            m[mask] = Sigma[0] * U[:, 0]

        if max_time is not None and time.time() - start > max_time:
            print("K-SVD: Stopping early because ran out of time.")
            break

        SM = (m.unsqueeze(1) @ M[ids]).squeeze(1)  # SM = S @ M
        error = torch.norm(SM - X)
        if error < 1e-4:
            print("K-SVD: Stopping early because error is near 0.")
            break
    print(f"K-SVD: error at {iter=}:", error.item())

    return M, ids, m

def randomized_round(tensor):
    """Perform randomized rounding on a tensor."""
    floor_val = torch.floor(tensor)
    prob = tensor - floor_val
    return floor_val + torch.bernoulli(prob).to(tensor.device)

def quantize(tensor, num_bits=8):
    qmin = -2**(num_bits - 1)
    qmax = 2**(num_bits - 1) - 1
    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = torch.round(qmin - min_val / scale)

    q_value = tensor / scale + zero_point
    q_value_rounded = randomized_round(q_value)
    q_tensor = q_value_rounded.clamp(qmin, qmax).char()  # Use torch.int8 for 8-bit
    return q_tensor, scale, zero_point

def dequantize(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

class SparseCodingEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, table, weights, h, x, sparse):
        ctx.save_for_backward(table, weights, h[x], h, x, torch.tensor([sparse]))

        vecs = table[h[x]]  # (batch_size, num_hashes, dim)
        weights_ = weights[x].unsqueeze(1)  # (batch_size, 1, num_hashes)
        return (weights_ @ vecs).squeeze(1)  # (batch_size, dim)

    @staticmethod
    def backward(ctx, grad_output):
        table, weights, hx, h, x, sparse = ctx.saved_tensors

        rows, dim = table.shape
        vocab, n_chunks = weights.shape
        bs, = x.shape
        assert h.shape == (vocab, n_chunks)
        assert hx.shape == (bs, n_chunks)
        assert grad_output.shape == (bs, dim)

        grad_table = torch.zeros_like(table)
        if sparse:
            grad_table = grad_table.to_sparse()
        assert weights[x].shape == (bs, n_chunks)
        wg = (weights[x].unsqueeze(2) @ grad_output.unsqueeze(1))
        assert wg.shape == (bs, n_chunks, dim)

        # The scatter code below is equivalent to the loop
        #
        #    for i in range(bs):
        #        for j in range(n_chunks):
        #            grad_table[h[x[i]][j]] += wg[i][j]
        #
        # Need to distribute, so grad_table[hx[i][j][k]] += wg[i][j][k]
        # We actually flatten the batch with the chunk dimension, since
        # it's all the same from the perspective of the gradients.
        grad_table.scatter_add_(0, hx.flatten()[:, None].expand(-1, dim), wg.flatten(0, 1))

        grad_weights = torch.zeros_like(weights)
        if sparse:
            grad_weights = grad_weights.to_sparse()
        assert table[hx].shape == (bs, n_chunks, dim)
        src = (table[hx] @ grad_output.unsqueeze(2)).squeeze(2) # (bs, 4)
        assert src.shape == (bs, n_chunks)

        # The scatter code below is equivalent to the loop:
        #
        #    for i in range(bs):
        #        grad_weights[x[i]] += src[i]
        #
        grad_weights.scatter_add_(0, x[:, None].expand(-1, n_chunks).to(torch.int64), src)

        # Returning gradients for table and weights.
        # Note that even though we are using scatter_add_, the gradient is still
        # a dense tensor, the sahpe of the tables themselves.
        return grad_table, grad_weights, None, None, None

def test_back():
    # TODO: Move this to unittests
    from torch.autograd import gradcheck

    # Initialize inputs
    vocab, bs, rows, dim, n_chunks = 6, 5, 3, 4, 2
    table = torch.rand(rows, dim, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(vocab, n_chunks, dtype=torch.float64, requires_grad=True)
    h = torch.randint(0, rows, (vocab, n_chunks), dtype=torch.long)
    x = torch.randint(0, vocab, (bs,), dtype=torch.long)
    sparse = False

    # Perform the gradcheck
    test = gradcheck(SparseCodingEmbeddingFunction.apply, (table, weights, h, x, sparse), eps=1e-6, atol=1e-4)
    print("Gradcheck Passed? ", test)
test_back()

class SparseCodingEmbedding(nn.Module):
    def __init__(
        self,
        num_params: int,
        vocab: int,
        dim: int,
        n_chunks: int,
        n_explore: int = 1,  # Number of random pointers per sample
        sparse: bool = False,
        num_bits: int = 8, # Number of bits per weight, None for infinite
        table_grad: bool = False,
    ):
        super().__init__()
        self.n_explore = n_explore
        self.sparse = sparse
        self.num_bits = num_bits
        rows = num_params // dim
        # Somehow doing requires_grad=False here works quite well...
        self.table = nn.Parameter(torch.empty(rows, dim), requires_grad=table_grad)
        self.weights = nn.Parameter(torch.empty(vocab, n_chunks))
        self.h = nn.Parameter(torch.empty((vocab, n_chunks), dtype=torch.int64), requires_grad=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.weights, -1, 1)
        self.h[:] = torch.randint(rows, size=self.h.shape)

    def forward(self, x):
        # Ideally this should be done at the time of updating the gradients, and we should
        # only dequantize the weights we actually need for the forward pass.
        # However, this should simulate the same effect.
        if self.num_bits is not None:
            with torch.no_grad():
                self.weights[:] = dequantize(*quantize(self.weights, self.num_bits))

        # Making our own forward/backward is about twice as fast.
        return SparseCodingEmbeddingFunction.apply(self.table, self.weights, self.h, x, self.sparse)

    @torch.no_grad()
    def cluster(self, k_svd_iters=100, sample_factor=200, verbose=False, max_time=None):
        rows, dim = self.table.shape
        vocab, n_chunks = self.h.shape

        # We use a sub-sampling strategy, similar to CCE
        n_samples = sample_factor * rows
        print(f'{vocab=}, {n_samples=}')

        if n_samples >= vocab:
            vecs = self.forward(torch.arange(vocab))
            s = n_chunks - self.n_explore
            M, indices, values = k_svd(vecs, self.table, s=s, n_iter=k_svd_iters, max_time=max_time)

            self.table[:] = M

            self.h[:, :s] = indices
            self.h[:, s:] = torch.randint(rows, size=(vocab, n_chunks - s), device=self.h.device)

            self.weights[:, :s] = values
            self.weights[:, s:] = 0

        else:
            # Pick a random set of indices from the vocab and do k_svd on those
            x = torch.from_numpy(np.random.choice(vocab, n_samples, replace=False))
            vecs = self.forward(x)
            s = n_chunks - self.n_explore
            M, _, _ = k_svd(vecs, self.table, s=s, n_iter=k_svd_iters, max_time=max_time)

            for j in range(0, vocab, n_samples):
                ids = torch.arange(j, min(j + n_samples, vocab))

                vecs = self.forward(ids)
                indices, values = omp(vecs, M, s)

                self.h[ids, :s] = indices
                self.h[ids, s:] = torch.randint(rows, size=(len(ids), n_chunks - s), device=self.h.device)

                self.weights[ids, :s] = values
                self.weights[ids, s:] = 0

        # Now to make it fair we have to compress the weights a bit more.
        # Like, we can use hashing like in hemb.
        # But what about the weights we learn (values)?
        # Maybe we can do a learned cluster on those as well?
        # But does that really make sense? Storing a pointer compared to just storing a quantized value?
        # But if I store a quantized value, I won't be able to train it afterwards?
        # I guess I could use some kind of super low bit representation?

        # Idea 1:
        #  - Compute all the "ideal" vectors, like in CCE
        #  - Run k-svd
        # - Make some random edges with 0 weight for exploration
        # Could even just run OMP? Well, I have to put something in the table.
        # If I'm just going to put the least-squares (MOD style), I might as well do k-svd.
        # And if I'm already doing k-svd, why not do a couple more iterations?
