import torch
from cce import sparse
from torch.autograd import gradcheck
import numpy as np


def generate_synthetic_data(n_samples, n_features, n_atoms, s):
    """
    Generate synthetic data for testing.
    """
    D_true = torch.randn(n_atoms, n_features)
    norms = torch.linalg.norm(D_true, axis=1) + 1e-6
    D_true /= norms[:, None]

    S_true = torch.zeros(n_samples, n_atoms)
    for i in range(n_samples):
        indices = torch.randint(0, n_atoms, (s,))
        values = torch.randn(s)
        S_true[i, indices] = values

    X = S_true @ D_true

    return X, D_true, S_true


def test_omp():
    n_samples, n_features, n_atoms, s = 100, 50, 20, 5
    X, D_true, S_true = generate_synthetic_data(n_samples, n_features, n_atoms, s)
    ids, m = sparse.omp(X, D_true, s)
    SM = (m.unsqueeze(1) @ D_true[ids]).squeeze(1)
    error = torch.norm(SM - X)
    assert error < 1e-2, "OMP reconstruction error is too high"


def old_omp(X, D, s):
    """
    X: data matrix of size (n_samples, n_features)
    D: dictionary of size (n_atoms, n_features)
    s: desired sparsity level
    Returns the sparse code matrix S of size (n_samples, n_atoms)
    """
    X = torch.from_numpy(X)
    D = torch.from_numpy(D)
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
    return S.numpy()


def old_k_svd(X, M, s, n_iter=50):
    n, d = X.shape
    k, _ = M.shape
    X = X.numpy()
    M = M.numpy()

    for iter in range(n_iter):
        # Sparse Coding
        S = old_omp(X, M, s)

        # Dictionary Update, one row at a time
        for j in range(k):
            # Find which samples are currently assumed to be using the kth atom
            I = np.where(S[:, j] != 0)[0]

            if len(I) == 0:
                continue

            # Compute current residuals
            E = X[I] - S[I] @ M
            # Add the current atom back in. This is like assuming it was
            # currently not used by any of the samples.
            E = E + np.outer(S[I, j], M[j])

            # SVD all the residuals.
            # We could use scipy or pytorch here to save time and only compute
            # the first singular vector
            U, Sigma, Vt = np.linalg.svd(E)
            M[j, :] = Vt[0, :]
            # We also update S, which is how k-svd can have an advantage over MOD
            S[I, j] = Sigma[0] * U[:, 0]
        print(f"K-SVD error at {iter}:", np.linalg.norm(S @ M - X))

    return torch.from_numpy(M), torch.from_numpy(S)


def test_k_svd():
    n_samples, n_features, n_atoms, s = 100, 20, 50, 5
    X, D_true, _ = generate_synthetic_data(n_samples, n_features, n_atoms, s)

    # Initialize with random dictionary
    M_init = torch.randn(n_atoms, n_features)

    # Ensure the dictionary has learned something by checking if the reconstruction error decreases
    ids_initial, m_initial = sparse.omp(X, M_init, s)
    SM_initial = (m_initial.unsqueeze(1) @ M_init[ids_initial]).squeeze(1)
    initial_error = torch.norm(SM_initial - X)

    M_learned, _, _ = sparse.k_svd(X, M_init, s, n_iter=10)
    # M_learned, _ = old_k_svd(X, M_init, s, n_iter=10)

    ids_learned, m_learned = sparse.omp(X, M_learned, s)
    SM_learned = (m_learned.unsqueeze(1) @ M_learned[ids_learned]).squeeze(1)
    learned_error = torch.norm(SM_learned - X)

    print(initial_error)
    print(learned_error)

    assert learned_error < initial_error, "K-SVD didn't reduce the reconstruction error"


def test_backprop():
    # Initialize inputs
    vocab, bs, rows, dim, n_chunks = 6, 5, 3, 4, 2
    table = torch.rand(rows, dim, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(vocab, n_chunks, dtype=torch.float64, requires_grad=True)
    h = torch.randint(0, rows, (vocab, n_chunks), dtype=torch.long)
    x = torch.randint(0, vocab, (bs,), dtype=torch.long)
    is_sparse = False

    # Perform the gradcheck
    assert gradcheck(sparse.SparseCodingEmbeddingFunction.apply, (table, weights, h, x, is_sparse), eps=1e-6, atol=1e-4)
