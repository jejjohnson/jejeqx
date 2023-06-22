import typing as tp
import equinox as eqx
from jaxtyping import Array


class Prior(eqx.Module):
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError()

    def loss(self, x: Array, x_gt: tp.Optional[Array] = None) -> Array:
        raise NotImplementedError()
