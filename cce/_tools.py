import torch
import numpy as np
import faiss
import time

use_sklearn = False


def batch_nn(X, Y, bs=None):
    if bs is None:
        bs = max(10**8 // len(Y), 1)
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
        _, indices = faiss.knn(X.cpu().numpy(), Y.cpu().numpy(), 1)
        return torch.from_numpy(indices).squeeze(1)
    else:
        _, indices = faiss.knn_gpu(res, X, Y, 1)
        return indices.squeeze(1)


class KMeans:
    """Simple wrapper for sklearn and faiss"""

    def __init__(self, n_clusters, n_iter, n_init, seed=None, verbose=False):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_init = n_init
        self.seed = seed if seed is not None else np.random.randint(2**16)
        self.verbose = verbose

    def fit(self, vecs):
        n, dim = vecs.shape
        if n <= self.n_clusters:
            # If we have more clusters centers than vectors, we put the vectors in
            # the first n rows, and put random points in the remaining rows.
            self.centroids = (torch.randn(self.n_clusters, dim) / dim**0.5).to(vecs.device)
            self.centroids[:n] = vecs
        elif use_sklearn:
            import sklearn.cluster

            kmeans = sklearn.cluster.KMeans(
                self.n_clusters,
                max_iter=self.n_iter,
                n_init=self.n_init,
                verbose=self.verbose,
                random_state=self.seed,
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
                seed=self.seed,
                # Disable warnings related to the points to n_cluster ratio
                min_points_per_centroid=1,
                max_points_per_centroid=n,
            )
            kmeans.train(vecs.detach().cpu().numpy())
            self.centroids = torch.from_numpy(kmeans.centroids).to(vecs.device)
        return self.centroids

    def find_nearest(self, bvecs):
        return batch_nn(bvecs, self.centroids)


def batch_rotary_nn(X, y, bs=None):
    len_x, k = X.shape
    (len_y,) = y.shape
    if bs is None:
        bs = max(10**8 // len(y), 1)
        print(f"{bs=}")
    y_extended = torch.cat((y, y))
    y2_cumsum = torch.cumsum(y_extended**2, dim=0)
    y2_cumsum = torch.cat((torch.zeros(1), y2_cumsum))
    y_norms = y2_cumsum[k : len_y + k] - y2_cumsum[:len_y]
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

            if not use_sklearn:
                labels = faiss_knn(vecs, centroids)
            else:
                labels = batch_nn(vecs, centroids)

            flat_labels = (labels[..., None] + torch.arange(self.dim, device=labels.device)).flatten(-2, -1) % len(
                table
            )

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
        # return batch_nn(vecs, self.centroids)
        return faiss_knn(vecs, self.centroids)
