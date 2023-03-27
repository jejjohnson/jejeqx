"""Code taken (and modified) from:
    https://github.com/lucidrains/siren-pytorch
"""
from typing import Optional, Literal, Union, Tuple, Callable
from jaxtyping import Float, Array
import math
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
from equinox import static_field
from equinox.nn.linear import Identity

PRNGKey = jax.random.PRNGKey


def sine_activation(x: Array, w0: float) -> Array:
    return w0 * x


def get_siren_init(dim: int, c: float, w0: float, is_first: bool):
    return (1 / dim) if is_first else (math.sqrt(c / dim) / w0)


class Sine(eqx.Module):
    """Sine activation function."""

    w0: float = eqx.static_field()

    def __init__(self, w0: float):
        """
        Args:
            w0 (int): the amplitude factor for the sine activation
        """
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)


class Siren(eqx.Module):
    """Performs a linear transformation.

    .. math::

        (a + b)^2 = a^2 + 2ab + b^2

    Continue!
    """

    w0: float = static_field()
    c: float = static_field()
    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = static_field()
    out_features: Union[int, Literal["scalar"]] = static_field()
    use_bias: bool = static_field()
    is_first: bool = static_field()
    activation: eqx.Module

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        is_first: bool = False,
        w0: float = 1.0,
        c: float = 6.0,
        activation: Optional[eqx.Module]=None,
        *,
        key: PRNGKey,
    ):
        """
        Args:
            in_features (Union[int, Literal[&quot;scalar&quot;]]): _description_
            out_features (Union[int, Literal[&quot;scalar&quot;]]): _description_
            key (PRNGKey): _description_
            use_bias (bool, optional): _description_. Defaults to True.
            is_first (bool, optional): _description_. Defaults to False.
            w0 (float, optional): _description_. Defaults to 1.0.
            c (float, optional): _description_. Defaults to 6.0.
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features

        lim = get_siren_init(dim=in_features_, c=c, w0=w0, is_first=is_first)

        self.weight = jrandom.uniform(
            wkey, (out_features_, in_features_), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.w0 = w0
        self.c = c
        self.is_first = is_first
        self.activation = Sine(self.w0) if activation is None else activation

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return self.activation(x)


class SirenNet(eqx.Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    .. math:: (a + b)^2 = a^2 + 2ab + b^2

    """

    layers: Tuple[Siren, ...]
    in_size: Union[int, Literal["scalar"]] = static_field()
    out_size: Union[int, Literal["scalar"]] = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        w0_initial: float = 30.0,
        w0: float = 1.0,
        c: float = 6.0,
        final_activation: Callable = Identity(),
        *,
        key: PRNGKey,
        **kwargs,
    ):
        """
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
            layers.append(Siren(in_size, out_size, w0=w0_initial, c=c, is_first=True, key=keys[0]))
        else:
            layers.append(Siren(in_size, width_size, w0=w0_initial, c=c, is_first=True, key=keys[0]))
            for i in range(depth - 1):
                layers.append(
                    Siren(width_size, width_size, w0=w0, c=c, key=keys[i + 1])
                )
            layers.append(Siren(width_size, out_size, w0=w0, c=c, activation=Identity(), key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth


    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return x
