import torch
import torch.nn as nn
import os
import time
import numpy as np


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
    norms = torch.linalg.norm(D, axis=1, keepdims=True) + 1e-6
    D2 = D / norms

    residuals = X.clone()
    ids = torch.zeros((n_samples, s), dtype=int)

    for step in range(s):
        corrs = torch.abs(residuals @ D2.T)
        ids[:, step] = torch.argmax(corrs, axis=1)
        subDs = D2[ids[:, : step + 1]].mT
        m = torch.linalg.lstsq(subDs, X, rcond=None).solution
        residuals = X - (subDs @ m.unsqueeze(2)).squeeze(2)

    S = torch.zeros((n_samples, n_atoms), dtype=m.dtype)
    r = torch.arange(n_samples, dtype=int).unsqueeze(-1)
    S[r, ids] = m
    S /= norms.T
    return S


def k_svd(X, M, s, n_iter, max_time=None):
    n, d = X.shape
    k, _ = M.shape

    if n_iter == 0:
        S = omp(X, M, s)
        return M, S

    start = time.time()
    for iter in range(n_iter):
        # Sparse Coding
        S = omp(X, M, s)

        # Dictionary Update, one row at a time
        for j in range(k):
            # Find which samples are currently assumed to be using the kth atom
            I = (S[:, j] != 0).nonzero(as_tuple=True)[0]

            if len(I) == 0:
                continue

            # Compute current residuals
            E = X[I] - torch.mm(S[I], M)
            # Add the current atom back in. This is like assuming it was
            # currently not used by any of the samples.
            E = E + torch.outer(S[I, j], M[j])

            # Use svd_lowrank from torch to save time and only compute
            # the first singular vector
            U, Sigma, V = torch.svd_lowrank(E, q=1)
            # U, Sigma, V = torch.linalg.svd(E)
            M[j, :] = V[0, :]
            # We also update S, which is how k-svd can have an advantage over MOD
            S[I, j] = Sigma[0] * U[:, 0]

        if max_time is not None and time.time() - start > max_time:
            print("Ran out of time")
            break
    print(f"K-SVD error at {iter}:", torch.norm(torch.mm(S, M) - X).item())

    # TODO: I don't really want S back. I just want the pointers (s per row) and the weights
    return M, S


def get_nonzero_indices_values(X):
    mask = X != 0
    rows = torch.arange(X.size(0), device=X.device).unsqueeze(-1).expand_as(X)
    indices = torch.where(
        mask, torch.arange(X.size(1), device=X.device), -torch.ones_like(X)
    ).long()
    values = torch.where(mask, X, torch.zeros_like(X))

    # Sort and filter out the placeholder (-1) values
    _, sorted_idx = indices.sort(dim=1, descending=True)
    range_idx = (
        torch.arange(X.size(0), device=X.device).unsqueeze(-1).expand_as(sorted_idx)
    )
    sorted_indices = torch.gather(indices, 1, sorted_idx)
    sorted_values = torch.gather(values, 1, sorted_idx)

    s = mask.sum(dim=1).max()
    return sorted_indices[:, :s], sorted_values[:, :s]


class SparseCodingEmbedding(nn.Module):
    def __init__(
        self,
        num_params: int,
        vocab: int,
        dim: int,
        n_chunks: int,
        n_explore: int = 2,  # Number of random pointers per sample
    ):
        super().__init__()
        rows = num_params // dim
        # Somehow doing requires_grad=False here works quite well...
        self.table = nn.Parameter(torch.empty(rows, dim),
                                  )

        self.weights = nn.Parameter(torch.empty(vocab, n_chunks),
                                  requires_grad=False
                                    )

        self.h = nn.Parameter(
            torch.randint(rows, size=(vocab, n_chunks)), requires_grad=False
        )

        self.n_explore = n_explore

        self.reset_parameters()

    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.weights, -1/5, 1/5)
        #nn.init.uniform_(self.weights, 1, 1)

    def forward(self, x):
        vecs = self.table[self.h[x]]  # (batch_size, num_hashes, dim)
        weights = self.weights[x].unsqueeze(1)  # (batch_size, 1, num_hashes)
        return (weights @ vecs).squeeze(1)  # (batch_size, dim)

    @torch.no_grad()
    def cluster(self, verbose=False, max_time=None):
        rows, dim = self.table.shape
        vocab, n_chunks = self.h.shape

        n_samples = vocab
        #n_samples = 200 * rows
        x = (
            torch.from_numpy(np.random.choice(vocab, n_samples, replace=False))
            if n_samples < vocab
            else torch.arange(vocab)
        )
        vecs = self.forward(x)

        s = n_chunks - self.n_explore
        M, S = k_svd(vecs, self.table, s=s, n_iter=1, max_time=max_time)
        indices, values = get_nonzero_indices_values(S)
        _, s2 = indices.shape
        if s2 != s:
            print(f"Warning: Only got {s2} (not {s}) non-zeros")
            s = s2

        self.table[:] = M

        self.h[:, :s] = indices
        self.h[:, s:] = torch.randint(rows, size=(vocab, n_chunks - s))

        self.weights[:, :s] = values
        #self.weights[:, s:] = 0
        nn.init.uniform_(self.weights[:, s:], -1/5, 1/5)

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
