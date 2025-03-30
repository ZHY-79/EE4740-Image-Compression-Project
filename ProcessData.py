import numpy as np

def Generate_Standard_Matrix_A(dim1, dim2):
    matrix_A = np.random.randn(dim1, dim2)
    column_norms = np.linalg.norm(matrix_A, axis=0)
    # column_norms = np.where(column_norms != 0, column_norms, 1)
    normalized_A = matrix_A / column_norms
    
    return normalized_A

def Generate_Standard_Matrix_A(dim1, dim2):

    # 方法1：直接生成随机向量并归一化
    matrix_A = np.random.randn(dim1, dim2)
    # 对每一列归一化
    column_norms = np.sqrt(np.sum(matrix_A**2, axis=0))
    # 避免除以零
    column_norms = np.where(column_norms > 0, column_norms, 1.0)
    normalized_A = matrix_A / column_norms
    
    return normalized_A


def Calculate_Sign_Matrix(A, x):
    
    return np.sign(A, x)

def Calculate_Sparse(x, threshold=1e-6):

    x = np.asarray(x).flatten()
    non_zero_count = np.sum(np.abs(x) > threshold) 
    return non_zero_count