import torch
import torch.nn as nn
import numpy as np

use_sklearn = True
if use_sklearn:
    import sklearn.cluster
else:
    import faiss


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
        self.table1 = nn.Parameter(torch.zeros(rows, n_chunks, chunk_size))
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
            if use_sklearn:
                kmeans = sklearn.cluster.KMeans(
                    rows, max_iter=niter, verbose=verbose, n_init=redo
                )
            else:
                kmeans = faiss.Kmeans(
                    chunk_size, rows, niter=niter, nredo=redo, verbose=verbose
                )

            for i in range(n_chunks):
                # We might as well do iid sampling for each column
                if n_samples < vocab:
                    x = torch.from_numpy(
                        np.random.choice(vocab, n_samples, replace=False)
                    )
                else:
                    x = torch.arange(vocab)

                # Compute current representations in the column and cluster them
                vecs = self.table0[self.h0[x, i], i] + self.table1[self.h1[x, i], i]
                vecs = vecs.detach().numpy()
                if use_sklearn:
                    kmeans.fit(vecs)
                    centroids = torch.from_numpy(kmeans.cluster_centers_)
                else:
                    kmeans.train(vecs)
                    centroids = torch.from_numpy(kmeans.centroids)

                # In the case where the vocab is really big, we decode it in batches.
                sums = torch.zeros(rows, chunk_size)
                counts = torch.zeros(rows)
                for j in range(0, vocab, n_samples):
                    ids = torch.arange(j, min(j + n_samples, vocab))

                    # Compute pre-clustering representation of batch
                    bvecs = (
                        self.table0[self.h0[ids, i], i]
                        + self.table1[self.h1[ids, i], i]
                    )

                    # Set the new h0 based on decoding he batch in the centroids
                    if use_sklearn:
                        I = kmeans.predict(bvecs.detach().numpy().astype(np.float32))
                        self.h0[ids, i] = torch.from_numpy(I).to(torch.long)
                    else:
                        _D, I = kmeans.index.search(bvecs.detach().numpy(), 1)
                        self.h0[ids, i] = torch.from_numpy(I[:, 0])

                    # Second table is initialized at random
                    self.h1[:, i] = torch.randint(rows, size=(vocab,))

                    # Compute residuals
                    resids = bvecs - centroids[self.h0[ids, i]]
                    sums.scatter_add_(0, self.h1[ids, i].unsqueeze(-1), resids)
                    counts += torch.bincount(self.h1[ids, i], minlength=rows).float()

                # Initialize the parameters from the centroids, to stabalize convergence
                self.table0[:, i, :] = centroids

                # Initialize table 1 using residuals. This is a arguably a tiny bit better
                # than 0 initialization.
                self.table1[:, i, :] = sums / counts.unsqueeze(-1)
                # self.table1[:, i, :] = 0
