import time
import torch
import torch.nn as nn
import numpy as np
from . import robe
import faiss


def batch_nn(X, Y, bs=None):
    if bs is None:
        bs = max(10**7 // len(Y), 1)
    nns = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for i in range(0, len(X), bs):
        dists = torch.cdist(X[i : i + bs], Y)
        nns[i : i + bs] = torch.argmin(dists, axis=1)
    return nns


def faiss_knn(X, Y):
    try:
        # Try to do it on GPU if available
        res = faiss.StandardGpuResources()
    except AttributeError:
        _, indices = faiss.knn(X.numpy(), Y.numpy(), 1)
        return torch.from_numpy(indices).squeeze(1)
    else:
        _, indices = faiss.knn_gpu(res, X, Y, 1)
        return indices.squeeze(1)


def batch_rotary_nn(X, y, bs=None):
    len_x, k = X.shape
    len_y, = y.shape
    if bs is None:
        bs = max(10**8 // len(y), 1)
        print(f'{bs=}')
    y_extended = torch.cat((y, y))
    y2_cumsum = torch.cumsum(y_extended ** 2, dim=0)
    y2_cumsum = torch.cat((torch.zeros(1), y2_cumsum))
    y_norms = y2_cumsum[k:len_y+k] - y2_cumsum[:len_y]
    # Test it
    # old_norms = torch.sum(rolling_window(y, k) ** 2, dim=1)
    # torch.testing.assert_close(y_norms, old_norms, rtol=1e-3, atol=1e-3)
    y_fft = torch.fft.rfft(torch.flip(y, [0]))
    nns = torch.zeros(len_x, dtype=torch.long, device=X.device)
    for i in range(0, len_x, bs):
        x = X[i : i + bs]
        x_norms = torch.sum(x**2, dim=1, keepdims=True)
        x_fft = torch.fft.rfft(x, n=len_y, dim=1)

        convolution = torch.fft.irfft(x_fft * y_fft[None], dim=1)
        ips = torch.flip(convolution, [1])
        dists = x_norms + y_norms[None, :] - 2 * ips

        # Compare with direct method
        # dists_old = torch.cdist(x, rolling_window(y, k)) ** 2
        # torch.testing.assert_close(dists, dists_old, atol=1e-3, rtol=1e-3)

        nns[i : i + bs] = torch.argmin(dists, axis=1)
    return nns


def rolling_window(x, dim):
    # Extend x to handle the wrap-around effect
    extended_x = torch.cat((x, x))
    # Create the rolling window view using stride tricks
    return extended_x.as_strided(size=(len(x), dim), stride=(1, 1))


class RotaryKMeans:
    def __init__(self, dim, n_iter, verbose=False):
        self.dim = dim
        self.n_iter = n_iter
        self.verbose = verbose
        self.centroids = None

    def fit(self, table, vecs, max_time=None):
        if len(table) >= len(vecs):
            self.centroids = rolling_window(table, self.dim)
            return table

        flat_vecs = vecs.flatten(-2, -1)

        fit_start = time.time()
        for i in range(self.n_iter):
            # We just use the previous table as initialization for kmeans.
            # Is that weird/bad?
            centroids = rolling_window(table, self.dim)

            labels = faiss_knn(vecs, centroids)
            #labels = batch_nn(vecs, centroids)

            flat_labels = (
                labels[..., None] + torch.arange(self.dim, device=labels.device)
            ).flatten(-2, -1) % len(table)

            table[:] = 0
            table.scatter_add_(0, flat_labels, flat_vecs)

            counts = torch.bincount(flat_labels, minlength=len(table))
            if self.verbose:
                print(f"Count std: {counts.float().std():.3}")

            table /= torch.clamp(counts, min=1).float()  # Avoid division by zero

            if max_time is not None and time.time() - fit_start >= max_time:
                print(f"Clustering ran out of time after {i+1} iterations.")
                break

        self.centroids = rolling_window(table, self.dim)
        return table

    def find_nearest(self, vecs):
        #return batch_nn(vecs, self.centroids)
        return faiss_knn(vecs, self.centroids)


class CCERobembedding(nn.Module):
    def __init__(
        self,
        vocab: int,
        size: int,
        chunk_size: int,
        n_chunks: int,
    ):
        super().__init__()
        self.vocab = vocab
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.size = size
        self.table0 = nn.Parameter(torch.empty(size))
        self.table1 = nn.Parameter(torch.empty(size))
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.chunk_size * self.n_chunks
        nn.init.uniform_(self.table0, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.table1, -(dim**-0.5), dim**-0.5)
        self.h0 = nn.Parameter(
            torch.randint(self.size, size=(self.vocab, self.n_chunks)),
            requires_grad=False,
        )
        self.h1 = nn.Parameter(
            torch.randint(self.size, size=(self.vocab, self.n_chunks)),
            requires_grad=False,
        )

    def forward(self, x):
        part0 = robe.get_slices(self.table0, self.h0[x], self.chunk_size)
        part1 = robe.get_slices(self.table1, self.h1[x], self.chunk_size)
        # Shape: (batch, chunk, dim)
        return (part0 + part1).flatten(1, 2)

    def cluster(self, niter=100, sample_factor=200, verbose=False, max_time=None):
        # The Faiss manual suggests you never need more than 200 samples per centroid
        n_samples = sample_factor * self.size

        with torch.no_grad():
            kmeans = RotaryKMeans(self.chunk_size, n_iter=niter, verbose=verbose)

            # We might as well do iid sampling for each column
            x = (
                torch.from_numpy(np.random.choice(self.vocab, n_samples, replace=False))
                if n_samples < self.vocab
                else torch.arange(self.vocab)
            )

            # Compute current representations in the column and cluster them
            part0 = robe.get_slices(self.table0, self.h0[x], self.chunk_size)
            part1 = robe.get_slices(self.table1, self.h1[x], self.chunk_size)

            # Flatten batch and chunk indicies, so we get a bunch of chunk_size
            # vectors.
            vecs = (part0 + part1).flatten(0, 1)
            new_table = kmeans.fit(self.table0, vecs, max_time=max_time)

            # We need to find the best index for everything in the vocab.
            # In the case where the vocab is really big, we decode it in batches.
            for j in range(0, self.vocab, n_samples):
                ids = torch.arange(j, min(j + n_samples, self.vocab))

                for i in range(self.n_chunks):
                    # Compute pre-clustering representation of batch
                    part0 = robe.get_slices(
                        self.table0, self.h0[ids, i], self.chunk_size
                    )
                    part1 = robe.get_slices(
                        self.table1, self.h1[ids, i], self.chunk_size
                    )
                    bvecs = part0 + part1

                    # Set the new h0 based on decoding he batch in the centroids
                    self.h0[ids, i] = kmeans.find_nearest(bvecs)

                    # Second table is initialized at random
                    self.h1[:, i] = torch.randint(self.size, size=(self.vocab,))

                # Initialize the parameters from the centroids, to stabalize convergence
                self.table0[:] = new_table

                # Initialize table1 to 0. We don't bother with initializing it to the
                # residuals, since that doesn't really help anything
                self.table1[:] = 0
