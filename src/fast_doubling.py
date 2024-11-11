def fast_doubling_fibonacci_iterative(n: int) -> int:
    """
    Computes the nth Fibonacci number using the Fast Doubling method (iterative).
    Handles arbitrary large integers.
    See reference here: https://arxiv.org/pdf/1012.0284
    Parameters:
        n (int): The index of the Fibonacci number to compute. Must be non-negative.

    Returns:
        int: The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("Fibonacci number is not defined for negative integers.")

    a, b = 0, 1
    m = n
    binary = bin(m)[2:]

    for bit in binary:
        c = a * (2 * b - a)
        d = a * a + b * b
        if bit == '1':
            a, b = d, c + d
        else:
            a, b = c, d
    return a