import torch
import torch.nn as nn
import numpy as np

use_sklearn = True
if use_sklearn:
    import sklearn.cluster
else:
    import faiss


def batch_nn(X, Y, bs=None):
    if bs is None:
        bs = max(10**8 // len(Y), 1)
    nns = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for i in range(0, len(X), bs):
        dists = torch.cdist(X[i : i + bs], Y)
        nns[i : i + bs] = torch.argmin(dists, axis=1)
    return nns


class KMeans:
    def __init__(self, n_clusters, n_iter, n_init, verbose=False):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_init = n_init
        self.verbose = verbose

    def fit(self, vecs):
        _, dim = vecs.shape
        if len(vecs) <= self.n_clusters:
            self.centroids = (
                torch.randn(self.n_clusters, vecs.shape[1]) / vecs.shape[1] ** 0.5,
            ).to(vecs.device)
            self.centroids[: len(vecs)] = vecs
        if use_sklearn:
            kmeans = sklearn.cluster.KMeans(
                self.n_clusters,
                max_iter=self.n_iter,
                n_init=self.n_init,
                verbose=self.verbose,
            )
            kmeans.fit(vecs.detach().cpu().numpy())
            self.centroids = torch.from_numpy(kmeans.cluster_centers_).to(vecs.device)
        else:
            kmeans = faiss.Kmeans(
                dim,
                self.n_clusters,
                niter=self.n_iter,
                nredo=self.n_init,
                verbose=self.verbose,
            )
            kmeans.train(vecs.detach().cpu().numpy())
            self.centroids = torch.from_numpy(kmeans.centroids).to(vecs.device)
        return self.centroids

    def find_nearest(self, bvecs):
        return batch_nn(bvecs, self.centroids)


class CCEmbedding(nn.Module):
    def __init__(
        self,
        vocab: int,
        rows: int,
        chunk_size: int,
        n_chunks: int,
    ):
        super().__init__()
        self.vocab = vocab
        self.table0 = nn.Parameter(torch.empty(rows, n_chunks, chunk_size))
        self.table1 = nn.Parameter(torch.empty(rows, n_chunks, chunk_size))
        self.reset_parameters()

    def reset_parameters(self):
        rows, n_chunks, chunk_size = self.table0.shape
        dim = chunk_size * n_chunks
        # Initializing match dlrm
        nn.init.uniform_(self.table0, -(dim**-0.5), dim**-0.5)
        nn.init.uniform_(self.table1, -(dim**-0.5), dim**-0.5)
        self.h0 = nn.Parameter(
            torch.randint(rows, size=(self.vocab, n_chunks)), requires_grad=False
        )
        self.h1 = nn.Parameter(
            torch.randint(rows, size=(self.vocab, n_chunks)), requires_grad=False
        )

    def forward(self, x):
        rows, n_chunks, chunk_size = self.table0.shape
        part0 = self.table0[self.h0[x], range(n_chunks)]
        part1 = self.table1[self.h1[x], range(n_chunks)]
        return (part0 + part1).flatten(1, 2)

    def cluster(self, niter=100, sample_factor=200, redo=1, verbose=False):
        rows, n_chunks, chunk_size = self.table0.shape
        vocab, _ = self.h0.shape

        # The Faiss manual suggests you never need more than 200 samples per centroid
        n_samples = sample_factor * rows

        with torch.no_grad():
            kmeans = KMeans(rows, n_iter=niter, n_init=redo, verbose=verbose)

            for i in range(n_chunks):
                # We might as well do iid sampling for each column
                x = (
                    torch.from_numpy(np.random.choice(vocab, n_samples, replace=False))
                    if n_samples < vocab
                    else torch.arange(vocab)
                )

                # Compute current representations in the column and cluster them
                vecs = self.table0[self.h0[x, i], i] + self.table1[self.h1[x, i], i]
                centroids = kmeans.fit(vecs)

                # In the case where the vocab is really big, we decode it in batches.
                # We just use n_samples as the batch_size.
                sums = torch.zeros(rows, chunk_size, device=self.table1.device)
                counts = torch.zeros(rows, device=self.table1.device)
                for j in range(0, vocab, n_samples):
                    ids = torch.arange(j, min(j + n_samples, vocab))

                    # Compute pre-clustering representation of batch
                    bvecs = (
                        self.table0[self.h0[ids, i], i]
                        + self.table1[self.h1[ids, i], i]
                    )

                    # Set the new h0 based on decoding he batch in the centroids
                    self.h0[ids, i] = kmeans.find_nearest(bvecs)

                    # Second table is initialized at random
                    self.h1[:, i] = torch.randint(rows, size=(vocab,))

                    # Compute residuals
                    resids = bvecs - centroids[self.h0[ids, i]]
                    sums.scatter_add_(0, self.h1[ids, i, None], resids)
                    counts += torch.bincount(self.h1[ids, i], minlength=rows).float()

                # Initialize the parameters from the centroids, to stabalize convergence
                self.table0[:, i, :] = centroids

                # Initialize table 1 using residuals. This is a arguably a tiny bit better
                # than 0 initialization.
                self.table1[:, i, :] = sums / counts.unsqueeze(-1)
