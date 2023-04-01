import equinox as eqx
import jax
from jaxtyping import Array


class Tanh(eqx.Module):
    """Tanh activation function."""

    def __call__(self, x: Array) -> Array:
        return jax.nn.tanh(x)
    
    
class ReLU(eqx.Module):
    
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)