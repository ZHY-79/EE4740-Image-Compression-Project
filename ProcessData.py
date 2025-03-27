import numpy as np

def Generate_Standard_Matrix_A(dim1, dim2):
    matrix_A = np.random.randn(dim1, dim2)
    column_norms = np.linalg.norm(matrix_A, axis=0)
    column_norms = np.where(column_norms != 0, column_norms, 1)
    normalized_A = matrix_A / column_norms
    
    return normalized_A

def Calculate_Sign_Matrix(A, x):
    
    return np.sign(A, x)