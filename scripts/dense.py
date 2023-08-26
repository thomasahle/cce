import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
import tqdm


def dense_cce(X, Y, k, num_iters=100):
    n, d1 = X.shape
    d2 = Y.shape[1]

    H = np.zeros((d1, d2 + k))
    M = np.zeros((d2 + k, d2))

    losses = []

    for i in tqdm.tqdm(range(num_iters)):
        T = H @ M
        losses.append(np.linalg.norm(X @ T - Y) ** 2)

        N = np.random.normal(size=(d1, k))
        H = np.hstack([T, N])

        M = np.linalg.lstsq(X @ H, Y, rcond=None)[0]

    return range(num_iters), losses


n, d1, k, d2 = 10**4, 10**3, 10**2, 10**1
X = np.random.randn(n, d1)
Y = np.random.randn(n, d2)

xs, ys = dense_cce(X, Y, k, num_iters=100)
plt.plot(xs, ys, label="Algorithm 1: Dense CCE for Least Squares")

T = np.linalg.lstsq(X, Y, rcond=None)[0]
a = np.linalg.norm(X @ T) ** 2
b = np.linalg.norm(X @ T - Y) ** 2

s = min(np.linalg.svd(X, compute_uv=False))
rho = s**2 / np.linalg.norm(X) ** 2
print(s, rho, np.linalg.norm(X))

zs = [(1 - rho) ** (i * k) * a + b for i in xs]
plt.plot(xs, zs, label="Theorem 3.1")

zs = [(1 - 1 / d1) ** (i * k) * a + b for i in xs]
plt.plot(xs, zs, label="$(1-1/d_1)^{ik}\|XT^*\|_F^2 + \|XT^* - Y\|_F^2$")

# zs = [np.exp(-i*k/d1) * a + b for i in xs]
# plt.plot(xs, zs, label='$e^{-ik/d_1}\|XT^*\|_F^2 + \|XT^* - Y\|_F^2$')

plt.xscale("log")
plt.xlabel("Iterations")
plt.ylabel(r"$\|XT_i - Y\|_F^2$")
plt.title(rf"Least squares with $d_1={d1}$, $k={k}$, $d_2={d2}$, $\rho={rho:.3}$")
plt.legend()
plt.show()
