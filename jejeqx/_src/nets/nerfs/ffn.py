from typing import Tuple, Optional
from jaxtyping import Array
from einops import repeat, rearrange
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


class RFFARD(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    ard: bool = eqx.static_field()

    def __init__(
        self,
        in_dim,
        num_features: int = 10,
        variance: float = 0.1,
        length_scale: float = 0.01,
        ard: bool = True,
        *,
        key=jrandom.PRNGKey(123),
    ):
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))

        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        if ard:
            length_scale = [length_scale] * in_dim

        self.log_variance = jnp.asarray(variance)
        self.log_length_scale = jnp.asarray(length_scale)
        self.num_features = num_features
        self.ard = ard

    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)

    @property
    def variance(self):
        return jnp.exp(self.log_variance)

    def __call__(self, x: Array, *, key=None) -> Array:

        x /= self.length_scale

        x = jnp.dot(self.omega, x)

        x = jnp.hstack([jnp.sin(x), jnp.cos(x)])

        x = rearrange(x, "... -> (...)")

        x *= jnp.sqrt(self.variance**2 / self.num_features)

        return x


class RFFARDCosine(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    ard: bool = eqx.static_field()

    def __init__(
        self,
        in_dim,
        num_features: int = 10,
        variance: float = 0.1,
        length_scale: float = 0.01,
        ard: bool = True,
        *,
        key=jrandom.PRNGKey(123),
    ):
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))

        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        if ard:
            length_scale = [length_scale] * in_dim
        self.log_variance = jnp.asarray(variance)
        self.log_length_scale = jnp.asarray(length_scale)
        self.num_features = num_features
        self.ard = ard

    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)

    @property
    def variance(self):
        return jnp.exp(self.log_variance)

    def __call__(self, x: Array, *, key=None) -> Array:

        x /= self.length_scale

        x = jnp.dot(self.omega, x) + self.bias

        x = jnp.cos(x)

        x = rearrange(x, "... -> (...)")

        x *= jnp.sqrt(self.variance**2 / self.num_features)

        return x


class RFFArcCosine(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    bias: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    ard: bool = eqx.static_field()

    def __init__(
        self,
        in_dim,
        num_features: int = 10,
        variance: float = 0.1,
        length_scale: float = 0.1,
        ard: bool = False,
        *,
        key=jrandom.PRNGKey(123),
    ):
        okey, bkey = jrandom.uniform(key=key, shape=2)
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))
        beta = jrandom.uniform(key=key, shape=(num_features,))

        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        self.log_variance = jnp.asarray(sigma)
        self.num_features = num_features
        if ard:
            length_scale = [length_scale] * in_dim
        self.log_length_scale = jnp.asarray(length_scale)
        self.ard = ard

    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)

    @property
    def variance(self):
        return jnp.exp(self.log_variance)

    def __call__(self, x: Array, *, key=None) -> Array:

        x /= self.length_scale

        x = jnp.where(x > 0, x, 0.0)

        x = rearrange(x, "... -> (...)")

        x *= jnp.sqrt(self.variance**2 / self.num_features)

        return x


class RFFLayer(eqx.Module):
    rff_layer: RFFARD
    linear_layer: eqx.nn.Linear

    def __init__(
        self,
        in_dim,
        out_dim,
        num_features: int = 10,
        variance: float = 0.1,
        length_scale: float = 0.01,
        use_bias: bool = True,
        method: str = "rbf",
        ard: bool = False,
        *,
        key=jrandom.PRNGKey(123),
    ):
        if method == "rbf_cos":
            self.rff_layer = RFFARDCosine(
                in_dim=in_dim,
                num_features=num_features,
                variance=variance,
                length_scale=length_scale,
                ard=ard,
                key=key,
            )
        elif method == "rbf":
            self.rff_layer = RFFARD(
                in_dim=in_dim,
                num_features=num_features,
                variance=variance,
                length_scale=length_scale,
                ard=ard,
                key=key,
            )
        elif method == "arcosine":
            self.rff_layer = RFFArcCosine(
                in_dim=in_dim,
                num_features=num_features,
                variance=variance,
                length_scale=length_scale,
                ard=ard,
                key=key,
            )
        else:
            raise ValueError(f"Unrecognized method: {method}")

        self.linear_layer = eqx.nn.Linear(
            in_features=self.rff_layer.out_dim,
            out_features=out_dim,
            use_bias=use_bias,
            key=key,
        )

    def __call__(self, x: Array, *, key=None) -> Array:

        x = self.rff_layer(x, key=key)
        x = self.linear_layer(x, key=key)

        return x


class RFFNet(eqx.Module):
    layers: Tuple[RFFLayer, ...]
    in_size: int = eqx.static_field()
    out_size: int = eqx.static_field()
    width_size: int = eqx.static_field()
    num_features: int = eqx.static_field()
    depth: int = eqx.static_field()
    ard: str = eqx.static_field()
    method: bool = eqx.static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        num_features: int = 64,
        ard: bool = False,
        method: str = "rbf",
        *,
        key=jrandom.PRNGKey(123),
    ):

        keys = jrandom.split(key, depth)

        layers = []

        if depth == 0:
            layers.append(
                RFFLayer(
                    in_dim=in_size,
                    out_dim=out_size,
                    num_features=num_features,
                    method=method,
                    ard=ard,
                    key=keys[0],
                )
            )
        else:

            layers.append(
                RFFLayer(
                    in_dim=in_size,
                    out_dim=width_size,
                    num_features=num_features,
                    method=method,
                    ard=ard,
                    key=keys[0],
                )
            )
            for i in range(depth - 2):
                layers.append(
                    RFFLayer(
                        in_dim=width_size,
                        out_dim=width_size,
                        num_features=num_features,
                        method=method,
                        ard=ard,
                        key=keys[i + 1],
                    )
                )
            layers.append(
                RFFLayer(
                    in_dim=width_size,
                    out_dim=out_size,
                    num_features=num_features,
                    method=method,
                    ard=ard,
                    key=keys[-1],
                )
            )

        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.num_features = num_features
        self.ard = ard
        self.method = method

    def __call__(self, x: Array, *, key: Optional[jrandom.PRNGKey] = None) -> Array:

        for layer in self.layers:
            x = layer(x)
        return x
