import torch
import torch.nn as nn
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
        self.dim = chunk_size * n_chunks
        self.table0 = nn.Parameter(torch.empty(rows, n_chunks, chunk_size))
        self.table1 = nn.Parameter(torch.zeros(rows, n_chunks, chunk_size))
        nn.init.uniform_(self.table0)
        self.h0 = nn.Parameter(torch.randint(rows, size=(vocab, n_chunks)), requires_grad=False)
        self.h1 = nn.Parameter(torch.randint(rows, size=(vocab, n_chunks)), requires_grad=False)

    def forward(self, x):
        rows, n_chunks, chunk_size = self.table0.shape
        part0 = self.table0[self.h0[x], range(n_chunks)]
        part1 = self.table1[self.h1[x], range(n_chunks)]
        return (part0 + part1).reshape(-1, self.dim)

    def cluster(self, niter=100, verbose=False):
        rows, n_chunks, chunk_size = self.table0.shape
        vocab, _ = self.h0.shape
        # The Faiss manual suggests you never need more than 200 samples per centroid
        n_samples = 200 * rows

        with torch.no_grad():
            kmeans = faiss.Kmeans(chunk_size, rows, niter=niter, verbose=verbose)
            for i in range(n_chunks):
                if n_samples < vocab:
                    a, b = torch.randint(rows, size=(2, n_samples))
                    vecs = self.table0[a, i] + self.table1[b, i]
                    kmeans.train(vecs.detach().numpy())
                    # In the case where the vocab is really big, we decode it in batches
                    for j in range(0, vocab, n_samples):
                        batch = torch.arange(j, min(j + n_samples, vocab))
                        vecs = self.table0[self.h0[batch, i], i] + self.table1[self.h1[batch, i], i]
                        vecs = vecs.detach().numpy()
                        _D, I = kmeans.index.search(vecs, 1)
                        self.h0[batch, i] = torch.from_numpy(I[:, 0])
                else:
                    # The simple case where we can just hash everything in one batch
                    x = torch.arange(vocab)
                    vecs = self.table0[self.h0[x, i], i] + self.table1[self.h1[x, i], i]
                    vecs = vecs.detach().numpy()
                    kmeans.train(vecs)
                    _D, I = kmeans.index.search(vecs, 1)
                    self.h0[:, i] = torch.from_numpy(I[:, 0])

                # Update tables
                self.table0[:, i, :] = torch.from_numpy(kmeans.centroids)
                self.table1[:, i] = 0

                # Rehash
                self.h1[:, i] = torch.randint(rows, size=(vocab,))

