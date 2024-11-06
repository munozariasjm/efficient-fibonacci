import jax.numpy as jnp
from jax import lax
import jax

jax.config.update("jax_enable_x64", True)

@jax.jit
def matrix_multiply(a, b):
    return jnp.matmul(a, b)

@jax.jit
def matrix_power(matrix, n):
    result = jnp.eye(2, dtype=matrix.dtype)
    def body_fun(val):
        result, matrix, n = val
        result = jax.lax.cond(
            n % 2 == 1,
            lambda _: matrix_multiply(result, matrix),
            lambda _: result,
            operand=None
        )
        matrix = matrix_multiply(matrix, matrix)
        n = n // 2
        return (result, matrix, n)

    def cond_fun(val):
        _, _, n = val
        return n > 0

    result, _, _ = jax.lax.while_loop(cond_fun, body_fun, (result, matrix, n))
    return result

@jax.jit
def fibonacci(n):
    def true_fun(_):
        return jnp.array(0, dtype=jnp.int64)
    def false_fun(n):
        matrix = jnp.array([[1, 1], [1, 0]], dtype=jnp.int64)
        powered_matrix = matrix_power(matrix, n - 1)
        return powered_matrix[0, 0]
    return jax.lax.cond(n == 0, true_fun, false_fun, operand=n)