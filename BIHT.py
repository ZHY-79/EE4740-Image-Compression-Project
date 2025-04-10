import numpy as np

# Keep only the top-k largest (by magnitude) entries in x, set others to zero
def hard_threshold(x, k):
    if k >= len(x):
        return x
    threshold = np.sort(np.abs(x))[-k]
    return x * (np.abs(x) >= threshold)

# Binary Iterative Hard Thresholding (BIHT) algorithm for sparse signal recovery
def biht(A, y, k, max_iter=10000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)  # Initial estimate
    tau = 0.001      # Step size for gradient update

    for iter in range(max_iter):
        # Estimate sign measurements from current x
        Ax = A @ x
        y_hat = np.sign(Ax)
        
        # Identify inconsistent measurements
        inconsistent = (y != y_hat)
        if not np.any(inconsistent):
            break

        # Gradient step (update based on inconsistency)
        a = x + (tau / 2) * A.T @ (y - y_hat)
        
        # Hard thresholding to keep only top-k components
        x = hard_threshold(a, k)
        
        # Normalize x to unit norm
        if np.linalg.norm(x) > 0:
            x = x / np.linalg.norm(x)
        
        # Stop if residual error is below tolerance
        residual = np.sum(inconsistent) / m
        if residual < tol:
            break

    return x
