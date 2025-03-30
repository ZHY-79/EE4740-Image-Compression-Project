import numpy as np
import math

def calculate_required_measurements(sparsity, n, constant_factor=1):

    # 检查输入参数
    if sparsity <= 0 or n <= 0:
        raise ValueError("稀疏度s和维度n都必须为正")
    if sparsity > n:
        raise ValueError("稀疏度s不能大于信号维度n")
    
    # 计算 s * log²(n/s)
    log_term = math.log2(n/sparsity)
    m = constant_factor * sparsity * (log_term ** 2)

    return math.ceil(m)

def compute_metrics(x_original, x_recovered):

    # 确保输入是numpy数组
    x_original = np.array(x_original)
    x_recovered = np.array(x_recovered)
    
    # 计算MSE (均方误差)
    # mse = np.mean((x_original - x_recovered) ** 2)
    
    # 归一化方向向量以消除幅度影响
    # x_original_norm = x_original / np.linalg.norm(x_original)
    # x_recovered_norm = x_recovered / np.linalg.norm(x_recovered)
    
    # nmse = np.mean((x_original_norm - x_recovered_norm) ** 2)
    
    # 论文的计算应该是这样的，而不是求均值
    # 计算MSE (向量差的L2范数平方)
    mse = np.linalg.norm(x_original - x_recovered) ** 2
    # 计算NMSE (归一化向量差的L2范数平方)
    x_original_norm = x_original / np.linalg.norm(x_original)
    x_recovered_norm = x_recovered / np.linalg.norm(x_recovered)
    
    nmse = np.linalg.norm(x_original_norm - x_recovered_norm) ** 2
    
    
    return mse, nmse