import equinox as eqx
from equinox import static_field
import jax
from jaxtyping import Array


class Tanh(eqx.Module):
    """Tanh activation function."""

    def __call__(self, x: Array) -> Array:
        return jax.nn.tanh(x)


class ReLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class Swish(eqx.Module):
    """Swish activation Function"""

    beta: float = static_field()

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, x: Array) -> Array:
        return x * jax.nn.sigmoid(self.beta * x)
