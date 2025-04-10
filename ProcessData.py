import numpy as np

# Generate a random matrix with normalized columns
def Generate_Standard_Matrix_A(dim1, dim2):
    matrix_A = np.random.randn(dim1, dim2)
    column_norms = np.linalg.norm(matrix_A, axis=0)
    normalized_A = matrix_A / column_norms
    return normalized_A

# Return the sign of each element in the matrix
def Calculate_Sign_Matrix(A, x):
    return np.sign(A)

# Count the number of elements with absolute value greater than a threshold
def Calculate_Sparse(x, threshold=1e-6):
    x = np.asarray(x).flatten()
    non_zero_count = np.sum(np.abs(x) > threshold) 
    return non_zero_count
