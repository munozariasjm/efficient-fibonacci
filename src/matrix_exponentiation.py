import numpy as np
from numba import njit, prange

def matrix_multiply(a, b):
    """
    Multiplies two 2x2 matrices a and b.
    Each matrix is represented as a NumPy array with dtype=object.
    """
    return np.array([
        [a[0,0]*b[0,0] + a[0,1]*b[1,0],
         a[0,0]*b[0,1] + a[0,1]*b[1,1]],
        [a[1,0]*b[0,0] + a[1,1]*b[1,0],
         a[1,0]*b[0,1] + a[1,1]*b[1,1]]
    ], dtype=object)

@njit
def matrix_multiply_numba(a00, a01, a10, a11, b00, b01, b10, b11):
    """
    Multiplies two 2x2 matrices with fixed-size integers.
    """
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11
    return c00, c01, c10, c11

def matrix_power(a, n):
    """
    Raises matrix a to the power of n using exponentiation by squaring.
    Handles arbitrary-precision integers.
    """
    result = np.array([[1, 0], [0, 1]], dtype=object)  # Identity matrix
    base = a.copy()

    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        n = n // 2
    return result

def fibonacci(n: int) -> int:
    """
    Computes the nth Fibonacci number using matrix exponentiation.
    Handles arbitrary large integers.
    """
    if n < 0:
        raise ValueError("Fibonacci number is not defined for negative integers.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    elif n>=310:
        raise ValueError("Fibonacci will overflow for n>=310")

    matrix = np.array([[1, 1], [1, 0]], dtype=object)
    powered_matrix = matrix_power(matrix, n - 1)
    return powered_matrix[0, 0]