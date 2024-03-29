import torch
import torch.nn as nn
import os
import time
import numpy as np
from torch.autograd import Function
from ._tools import faiss_knn
from .hash import PolyHash
from itertools import count
from tqdm import tqdm
from ortools.linear_solver import pywraplp
import random


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
    ids = torch.zeros((n_samples, s), dtype=int, device=X.device)

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
    last_error = 10**10
    for iter in range(n_iter):
        # Sparse Coding
        ids, m = omp(X, M, s)

        # Dictionary Update, one row at a time
        for j in range(k):
            # Find which samples are currently assumed to be using the kth atom
            mask = ids == j
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
            M[j, :] = V[:, 0]
            # We also update S, which is how k-svd can have an advantage over MOD
            # S[I, j] = Sigma[0] * U[:, 0]
            m[mask] = Sigma[0] * U[:, 0]

        SM = (m.unsqueeze(1) @ M[ids]).squeeze(1)  # SM = S @ M
        error = torch.norm(SM - X)
        if error < 1e-4:
            print("K-SVD: Stopping early because error is near 0.")
            break

        if error > last_error:
            print(f"K-SVD: Stopping early because error is growing. last_error={last_error.item()}.")
            break
        last_error = error

        if max_time is not None and time.time() - start > max_time:
            print("K-SVD: Stopping early because ran out of time.")
            break

    print(f"K-SVD: error at {iter=}:", error.item())

    return M, ids, m


def mini_ksvd(M, s, batch_maker, n_batches, n_iter, max_time=None):
    # Inspired by https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    assert n_iter >= 1
    k, _ = M.shape

    start = time.time()
    last_error = 10**10
    for iter in range(n_iter):
        error = 0
        new_Ms = torch.zeros_like(M)
        n = 0
        for j in tqdm(range(n_batches)):
            X = batch_maker(j)
            if X is None:
                break
            new_M = M.clone()
            # Sparse coding
            ids, m = omp(X, new_M, s)
            # Find new dictionary
            for j in range(k):
                mask = ids == j
                I = torch.any(mask, dim=1)
                if not torch.any(I):
                    continue
                E = X[I] - (m[I].unsqueeze(1) @ new_M[ids[I]]).squeeze(1) + torch.outer(m[mask], new_M[j])

                # Could use the cheaper alternative decomposition in Approximate K-SVD here
                U, Sigma, V = torch.svd_lowrank(E, q=1)
                new_M[j, :] = V[:, 0]
                m[mask] = Sigma[0] * U[:, 0]

            # "Gradient" update
            # M = (1-lr)*M + lr*new_M
            # M = new_M
            new_Ms += new_M
            n += 1

            SM = (m.unsqueeze(1) @ M[ids]).squeeze(1)  # SM = S @ M
            error += torch.norm(SM - X) ** 2

        M = new_Ms / n

        error = error**0.5
        if error < 1e-4:
            print("K-SVD: Stopping early because error is near 0.")
            break

        if error > last_error:
            print(f"K-SVD: Stopping early because error is growing. last_error={last_error.item()}.")
            break
        last_error = error

        if max_time is not None and time.time() - start > max_time:
            print("K-SVD: Stopping early because ran out of time.")
            break

    print(f"K-SVD: error at {iter=}:", error.item())

    return M


def randomized_round(tensor):
    """Perform randomized rounding on a tensor."""
    floor_val = torch.floor(tensor)
    prob = tensor - floor_val
    return floor_val + torch.bernoulli(prob).to(tensor.device)


def quantize(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    q_value = tensor / scale + zero_point
    q_value_rounded = randomized_round(q_value)
    assert num_bits <= 8, "Can't cast to int8 if num_bits > 8"
    q_tensor = q_value_rounded.clamp(qmin, qmax).char()  # Use torch.int8 for 8-bit
    return q_tensor, scale, zero_point


def dequantize(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)


def solveAssignmentProblem(N, M, price_function):
    print(f"Making LP problem of size {N} times {M}...")
    solver = pywraplp.Solver.CreateSolver("GLOP")  # Use 'GLOP' for LP

    # Create variables
    x = {}
    for i in range(N):
        for j in range(M):
            x[i, j] = solver.BoolVar(f"x[{i}][{j}]")

    # Set objective
    objective = solver.Objective()
    for i in range(N):
        for j in range(M):
            objective.SetCoefficient(x[i, j], price_function(i, j))
    objective.SetMinimization()

    # Add constraints
    for i in range(N):
        solver.Add(sum(x[i, j] for j in range(M)) == 1)

    for j in range(M):
        solver.Add(sum(x[i, j] for i in range(N)) == N / M)

    print("Running solver...")
    # Solve the model
    status = solver.Solve()

    assignments = []
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(N):
            for j in range(M):
                if int(x[i, j].solution_value()) == 1:
                    assignments.append(j)
    else:
        print("The problem does not have an optimal solution!")

    return assignments


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
        (bs,) = x.shape
        assert h.shape == (vocab, n_chunks)
        assert hx.shape == (bs, n_chunks)
        assert grad_output.shape == (bs, dim)

        grad_table = torch.zeros_like(table)
        if sparse:
            grad_table = grad_table.to_sparse()
        assert weights[x].shape == (bs, n_chunks)
        wg = weights[x].unsqueeze(2) @ grad_output.unsqueeze(1)
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
        src = (table[hx] @ grad_output.unsqueeze(2)).squeeze(2)  # (bs, 4)
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


class SparseCodingEmbedding(nn.Module):
    def __init__(
        self,
        num_params: int,
        vocab: int,
        dim: int,
        n_chunks: int,
        n_explore: int = 1,  # Number of random pointers per sample
        sparse: bool = False,
        num_bits: int = 8,  # Number of bits per weight, None for infinite
        table_grad: bool = True,
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
        self.h[:] = torch.randint(rows, size=self.h.shape, device=self.h.device)

    def forward_old(self, x):
        vecs = self.table[self.h[x]]  # (batch_size, num_hashes, dim)
        weights_ = self.weights[x].unsqueeze(1)  # (batch_size, 1, num_hashes)
        return (weights_ @ vecs).squeeze(1)  # (batch_size, dim)

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
        print(f"{vocab=}, {n_samples=}")

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

                # We could compute the reconstruction error here, and the gradient,
                # to update M with gradient descent. Could be a separate loop as well.

                self.h[ids, :s] = indices
                self.h[ids, s:] = torch.randint(rows, size=(len(ids), n_chunks - s), device=self.h.device)

                self.weights[ids, :s] = values
                self.weights[ids, s:] = 0

        # Alternatively we could go through each batch and update the dictionary
        # using online dictionary learning.

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


class SparseCodingEmbedding2(nn.Module):
    # Whemb verrsion of SparseCodingEmbedding.
    # Instead of using quantized weights, it simply "hashes" the weights to the
    # most similar existing parameters in the table.

    def __init__(
        self,
        num_params: int,
        vocab: int,
        dim: int,
        n_chunks: int,
        sparse=False,
    ):
        super().__init__()
        self.sparse = sparse
        self.n_chunks = n_chunks
        rows = num_params // dim
        assert rows >= n_chunks, "Must have at least as many rows as chunks"
        self.table = nn.Parameter(torch.empty(rows, dim))
        self.h0 = nn.Parameter(torch.empty((vocab, n_chunks), dtype=torch.int64), requires_grad=False)
        self.h1 = nn.Parameter(torch.empty((vocab, n_chunks), dtype=torch.int64), requires_grad=False)
        # Polyhash somehow works a bit better than a table hash (above). Nobody knows why.
        # self.h0 = PolyHash(num_hashes=n_chunks, output_range=rows)
        # self.h1 = PolyHash(num_hashes=n_chunks, output_range=rows * dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        rows, dim = self.table.shape
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)
        self.h0[:] = torch.randint(self.table.shape[0], size=self.h0.shape, device=self.h0.device)
        self.h1[:] = torch.randint(self.table.numel(), size=self.h1.shape, device=self.h1.device)

    def forward(self, x):
        rows, dim = self.table.shape
        vecs = self.table[self.h0[x]]  # (batch_size, num_hashes, dim)
        weights = self.table.flatten()[self.h1[x]].unsqueeze(1) * dim**0.5 * self.n_chunks**-0.5
        # vecs = self.table[self.h0(x)]  # (batch_size, num_hashes, dim)
        # weights = self.table.flatten()[self.h1(x)].unsqueeze(1) * dim**.5 * self.n_chunks**-.5
        return (weights @ vecs).squeeze(1)  # (batch_size, dim)

    def _find_batch_size(self):
        vocab, n_chunks = self.h0.shape
        last = np.inf
        print("Finding fastest batch size")
        for log_batch_size in count(5):
            print(log_batch_size, "...")
            batch_size = 2**log_batch_size

            def make_batch(j):
                if j * batch_size >= vocab:
                    return None
                x = torch.arange(j * batch_size, min((j + 1) * batch_size, vocab))
                return self.forward(x)

            start = time.time()
            _ = mini_ksvd(self.table, n_chunks, make_batch, vocab // batch_size + 1, n_iter=1)
            elapsed = time.time() - start
            print(batch_size, elapsed)
            if elapsed > last:
                break
            last = elapsed
        return 2 ** (log_batch_size - 1)

    @torch.no_grad()
    def cluster(self, k_svd_iters=100, batch_size=None, verbose=False, max_time=None):
        rows, dim = self.table.shape
        vocab, n_chunks = self.h0.shape
        n_atoms = self.table.numel()

        # max_time *= 10

        if batch_size is None:
            # batch_size = 10**9 // rows
            batch_size = 10**8 // rows
            # batch_size = self._find_batch_size()
            # batch_size = max(2 * rows, 4096*2)

        print(f"{batch_size=}, {vocab=}, {rows=}")

        if batch_size >= vocab:
            vecs = self.forward(torch.arange(vocab))
            M, indices, values = k_svd(vecs, self.table, s=n_chunks, n_iter=k_svd_iters, max_time=max_time)
            self.table[:] = M

            indices, values = omp(vecs, M, n_chunks)
            self.h0[:] = indices

            labels = faiss_knn(values.reshape(-1, 1), M.reshape(-1, 1) * dim**0.5)
            self.h1[:] = labels.reshape(vocab, n_chunks).to(self.h1.device)

            cnts = torch.bincount(labels, minlength=n_atoms)

        else:
            # Pick a random set of indices from the vocab and do k_svd on those
            # x = torch.from_numpy(np.random.choice(vocab, n_samples, replace=False))
            n_batches = vocab // batch_size + 1

            def make_batch(j):
                if j * batch_size >= vocab:
                    return None
                x = torch.arange(j * batch_size, min((j + 1) * batch_size, vocab))
                return self.forward(x)

            M = mini_ksvd(self.table, n_chunks, make_batch, n_batches, n_iter=k_svd_iters, max_time=max_time)
            self.table[:] = M

            cnts = torch.zeros(n_atoms)

            for j in range(0, vocab, batch_size):
                ids = torch.arange(j, min(j + batch_size, vocab))

                vecs = self.forward(ids)
                indices, values = omp(vecs, M, n_chunks)
                self.h0[ids] = indices

                flatvals = values.flatten()
                flattabs = M.flatten() * dim**0.5
                # def price_function(vi, tj):
                #     return (flatvals[vi] - flattabs[vj])**2
                # labels = solveAssignmentProblem(len(flatvals), len(flattabs), price_function)

                start = time.time()
                sorted_ = flattabs.sort()
                # For each value, this finds the table element that's either higher or lower (index-1)
                ind = torch.searchsorted(sorted_.values, flatvals)
                high = torch.minimum(ind, torch.tensor([len(flattabs) - 1]))
                low = torch.maximum(ind - 1, torch.tensor([0]))
                labels = torch.stack([sorted_.indices[low], sorted_.indices[high]], dim=1)
                choice = torch.argmin((flattabs[labels] - flatvals[:, None]) ** 2, dim=1)
                labels0 = labels[range(len(labels)), choice]
                # print(time.time()-start)

                ps = (labels[:, 1] - flatvals) / (labels[:, 1] - labels[:, 0])
                choice = torch.where(torch.rand(len(labels)) < ps, 1, 0)
                labels1 = labels[range(len(labels)), choice]

                start = time.time()
                labels2 = faiss_knn(values.reshape(-1, 1), M.reshape(-1, 1) * dim**0.5)
                # print(time.time()-start)

                # print(labels)
                # print(labels1)
                # for i in range(3):
                # print(flatvals[i], flattabs[labels0[i]], flattabs[labels1[i]], flattabs[labels2[i]])

                labels = labels2
                # 0: .789
                # 1: .779
                # 2: .79

                self.h1[ids] = labels.reshape(len(ids), n_chunks).to(self.h1.device)

                cnts += torch.bincount(labels, minlength=n_atoms)

        # Measure whether weight pointers are well spread out
        ps = cnts / cnts.sum()
        ent = (ps * torch.log(1 / ps)).nansum().item()
        uniform = torch.tensor([1 / n_atoms] * n_atoms)
        maxent = (uniform * torch.log(1 / uniform)).nansum().item()
        print(f"Value label entropy: {ent/maxent*100:.1f}% ({ent:.3} out of {maxent:.3})")
