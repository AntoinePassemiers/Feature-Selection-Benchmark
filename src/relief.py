import numpy as np
import scipy.spatial


def relief(X, y, n_iter=None, eps=1e-15):
    D = scipy.spatial.distance_matrix(X, X)
    scores = np.zeros(X.shape[1])
    ranges = np.max(X, axis=0) - np.min(X, axis=0)
    ranges = np.clip(ranges, eps, None)
    if n_iter is None:
        n_iter = X.shape[0]
    mm = np.random.choice(X.shape[0], n_iter, replace=False)
    for i in mm:
        n_hit = None
        n_miss = None
        for j in range(X.shape[0]):
            if i == j:
                continue
            if y[j] == y[i]:
                if n_hit is None:
                    n_hit = j
                elif D[i, j] < D[i, n_hit]:
                    n_hit = j
            else:
                if n_miss is None:
                    n_miss = j
                elif D[i, j] < D[i, n_miss]:
                    n_miss = j
        scores += (-np.abs(X[i, :] - X[n_hit, :]) + np.abs(X[i, :] - X[n_miss, :])) / ranges
    scores /= n_iter
    return scores
