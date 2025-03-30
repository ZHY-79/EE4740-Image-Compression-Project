import numpy as np

def hard_threshold(x, k):
    if k >= len(x):
        return x
    threshold = np.sort(np.abs(x))[-k]
    return x * (np.abs(x) >= threshold)

def biht(A, y, k, max_iter=10000, tol=1e-6, tau=0.01):
    m, n = A.shape
    x = np.zeros(n)  # 初始估计
    
    for iter in range(max_iter):
        # 计算当前估计的符号测量
        Ax = A @ x
        y_hat = np.sign(Ax)
        
        # 检查一致性
        inconsistent = (y != y_hat)
        if not np.any(inconsistent):
            break
        
        # 梯度下降步骤 (公式13)
        a = x + (tau/2) * A.T @ (y - y_hat)
        
        # 硬阈值步骤 (公式14)
        x = hard_threshold(a, k)
        
        # 每步都归一化
        if np.linalg.norm(x) > 0:
            x = x / np.linalg.norm(x)
        
        # 监控收敛
        residual = np.sum(inconsistent) / m
        if residual < tol:
            break
    
    return x

def biht_hinge(A, y, k, max_iter=1000, tol=1e-6, tau=0.5, kappa=1.0):
    m, n = A.shape
    x = np.zeros(n)  # 初始估计
    
    for iter in range(max_iter):
        # 计算当前估计的投影
        Ax = A @ x
        y_hat = np.sign(Ax)
        
        # 检查一致性
        inconsistent = (y != y_hat)
        if not np.any(inconsistent):
            break
        
        # 铰链损失梯度下降步骤
        a = x - tau * A.T @ ((y_hat - kappa) - 1) / 2
        
        # 硬阈值步骤
        x = hard_threshold(a, k)
        
        # 归一化
        if np.linalg.norm(x) > 0:
            x = x / np.linalg.norm(x)
        
        # 监控收敛
        residual = np.sum(inconsistent) / m
        if residual < tol:
            break
    
    return x

def biht_l1(A, y, k, max_iter=10000, tol=1e-6, mu=1.0):
    m, n = A.shape
    z = np.zeros(n)
    
    for iter in range(max_iter):
        Az = A @ z
        y_hat = np.sign(Az)
        
        inconsistent = (y != y_hat)
        if not np.any(inconsistent):
            break
            
        residual_vec = y * (np.maximum(0, 1 - y * Az))
        grad = A.T @ residual_vec
        p = z + mu * grad
        
        z = hard_threshold(p, k)
        
        if np.linalg.norm(z) > 0:
            z = z / np.linalg.norm(z)
        
        residual = np.sum(inconsistent) / m
        if residual < tol:
            break
    
    return z