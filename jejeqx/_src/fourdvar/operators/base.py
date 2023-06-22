import typing as tp
import equinox as eqx
import jax.numpy as jnp
from jejeqx._src.fourdvar.operators.identity import Identity
from jaxtyping import Array


class ObsOperator(eqx.Module):
    operator: tp.Callable

    def __init__(self, operator: tp.Callable = Identity):
        self.operator = operator

    def __call__(self, x: Array) -> Array:
        return self.operator(x)

    def loss(self, x: Array, y: Array, mask: tp.Optional[Array] = None) -> Array:
        # nans to numbers
        y = jnp.nan_to_num(y)

        if mask is not None:
            loss = jnp.sum(mask * (x - y) ** 2)
        else:
            loss = jnp.sum((x - y) ** 2)

        return loss
