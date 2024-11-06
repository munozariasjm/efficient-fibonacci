from ..src.matrix_exponentiation import fibonacci
import pytest

n_range = range(50)

# Test cases
def test_sequence(n):
    assert fibonacci(n) + fibonacci(n + 1) == fibonacci(n + 2)


@pytest.mark.parametrize("n", n_range)
def test_sequence_1(n):
    test_sequence(n)