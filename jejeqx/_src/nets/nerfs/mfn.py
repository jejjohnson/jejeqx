"""Code taken (and modified) from:
    https://github.com/boschresearch/multiplicative-filter-networks
"""
from typing import Optional, Literal, Union, Tuple, Callable
from jaxtyping import Float, Array
import math
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
from equinox import static_field
from equinox.nn import Identity, Linear
from jejeqx._src.nets.activations import ReLU

PRNGKey = jax.random.PRNGKey


class MFNBase(eqx.Module):
    layers: Tuple[Linear, ...]
    in_size: Union[int, Literal["scalar"]] = static_field()
    out_size: Union[int, Literal["scalar"]] = static_field()
    width_size: int = static_field()
    depth: int = static_field()
    final_activation: eqx.Module

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        w0: float = 1.0,
        use_bias: bool = True,
        final_activation: Callable = None,
        *,
        key: PRNGKey = jrandom.PRNGKey(123),
        **kwargs,
    ):
        """
        /gpfswork/rech/yrf/commun/data_challenges/dc21a_ose/test
        w1OgWd
        johnsonj@univ-grenoble-alpes.fr
        Args:
            in_size (int): The input size. The input to the module should be a vector of
                shape (in_features,)
            out_size (int): The output size. The output from the module will be a vector
                of shape (out_features,).
            width_size (int): The size of each hidden layer.
            depth (int): The number of hidden layers.
            activation (Callable): The activation function after each hidden layer. Defaults to
                ReLU.
            final_activation (Callable): The activation function after the output layer. Defaults
                to the identity.
            key (KEY): A (jax.random.PRNGKey) used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.
        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                Linear(
                    in_features=width_size,
                    out_features=out_size,
                    use_bias=use_bias,
                    key=keys[0],
                )
            )
        else:
            for i in range(depth - 1):
                layers.append(
                    Linear(
                        in_features=width_size,
                        out_features=width_size,
                        use_bias=use_bias,
                        key=keys[i + 1],
                    )
                )
            layers.append(
                Linear(
                    in_features=width_size,
                    out_features=out_size,
                    use_bias=use_bias,
                    key=keys[-1],
                )
            )

        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth

        self.final_activation = (
            Identity() if final_activation is None else final_activation
        )

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        out = self.filters[0](x)
        for ifilter, ilayer in zip(self.filters[1:], self.layers[:-1]):
            out = ilayer(out) * ifilter(x)
        out = self.layers[-1](out)
        out = self.final_activation(out)
        return out


class FourierLayer(eqx.Module):
    linear: Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        weight_scale: float = 1.0,
        *,
        key=jrandom.PRNGKey(123),
    ):

        key, bkey = jrandom.split(key, 2)
        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            key=key,
        )

        # modify weights and biases
        self.linear = eqx.tree_at(
            lambda x: x.weight, self.linear, weight_scale * self.linear.weight
        )
        bias = jrandom.uniform(bkey, (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        self.linear = eqx.tree_at(lambda x: x.bias, self.linear, bias)

    def __call__(self, x, *, key=None) -> Array:

        return jnp.sin(self.linear(x=x, key=key))


class FourierNet(MFNBase):
    filters: Tuple[FourierLayer, ...]

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int = 2,
        w0: float = 1.0,
        input_scale: float = 256,
        use_bias: bool = True,
        final_activation: Callable = None,
        *,
        key: PRNGKey = jrandom.PRNGKey(123),
        **kwargs,
    ):
        *fkeys, key = jrandom.split(key, depth + 2)
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            w0=w0,
            use_bias=use_bias,
            final_activation=final_activation,
            key=key,
        )

        filters = list()
        for i in range(depth + 1):

            filters.append(
                FourierLayer(
                    in_features=in_size,
                    out_features=width_size,
                    use_bias=use_bias,
                    weight_scale=input_scale / jnp.sqrt(depth + 1),
                    key=fkeys[i],
                ),
            )

        self.filters = tuple(filters)


class GaborLayer(eqx.Module):
    mu: Array
    gamma: Array
    linear: Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        weight_scale: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        key=jrandom.PRNGKey(123),
    ):
        gkeys = jrandom.split(key, 4)
        self.mu = (
            2 * jrandom.normal(key=gkeys[0], shape=(in_features, out_features)) - 1
        )
        self.gamma = jrandom.gamma(key=gkeys[1], a=alpha, shape=(out_features,))

        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            key=gkeys[2],
        )

        # modify weights and biases
        self.linear = eqx.tree_at(
            lambda x: x.weight, self.linear, weight_scale * self.linear.weight
        )
        bias = jrandom.uniform(gkeys[3], (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        self.linear = eqx.tree_at(lambda x: x.bias, self.linear, bias)

    def __call__(self, x, *, key=None) -> Array:

        # print(x.shape, self.mu.shape)
        A = jnp.sum(x**2, axis=-1)[..., None]
        B = jnp.sum(self.mu**2, axis=0)[None, :]
        # print(A.shape, B.shape)
        # print(x.shape, self.mu.T.shape)
        # C = jnp.einsum("i,ij->j", x, self.mu)
        C = -2 * x @ self.mu
        # print("A, B, C:", A.shape, B.shape, C.shape)

        D = (A + B + C).squeeze()
        # print("here:", D.shape)
        # print("D:", D.shape)
        # a_min_b = x[..., None]-self.mu

        # D = jnp.sqrt(jnp.einsum("ij,ij->j", a_min_b, a_min_b))

        return jnp.sin(self.linear(x=x, key=key)) * jnp.exp(-0.5 * D * self.gamma)


class GaborNet(MFNBase):
    filters: Tuple[GaborLayer, ...]

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int = 2,
        w0: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        input_scale: float = 256,
        use_bias: bool = True,
        final_activation: Callable = None,
        *,
        key: PRNGKey = jrandom.PRNGKey(123),
        **kwargs,
    ):
        *fkeys, key = jrandom.split(key, depth + 2)
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            w0=w0,
            use_bias=use_bias,
            final_activation=final_activation,
            key=key,
        )

        filters = list()
        for i in range(depth + 1):

            filters.append(
                GaborLayer(
                    in_features=in_size,
                    out_features=width_size,
                    use_bias=use_bias,
                    weight_scale=input_scale / jnp.sqrt(depth + 1),
                    alpha=alpha / (depth + 1),
                    beta=beta,
                    key=fkeys[i],
                ),
            )

        self.filters = tuple(filters)
