from jaxtyping import Array
from einops import repeat, rearrange
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


class CoordEncoding(eqx.Module):
    projection: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()

    def __init__(self, projection):
        self.in_dim = projection.shape[0]
        self.out_dim = 2 * projection.shape[1]
        self.projection = projection

    def __call__(self, x: Array, *, key=None) -> Array:

        x = repeat(x, "... -> ... 1")

        x = jnp.dot(x, self.projection)

        x = 2 * jnp.pi * x

        x = jnp.hstack([jnp.sin(x), jnp.cos(x)])

        x = rearrange(x, "... -> (...)")

        return x


class IdentityEncoding(eqx.Module):
    projection: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()

    def __init__(self, in_dim: int):
        projection = jnp.eye(in_dim)
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.projection = projection

    def __call__(self, x: Array, *, key=None):
        return x


class SinusoidalEncoding(eqx.Module):
    projection: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()

    def __init__(self, in_dim, num_features: int = 10):
        projection = sincos_freq(num_features)

        projection = repeat(projection, "F -> D F", D=in_dim)

        self.in_dim = projection.shape[0]
        self.projection = projection
        self.out_dim = num_features * 2 * in_dim

    def __call__(self, x: Array, *, key=None) -> Array:

        x = repeat(x, "... -> ... 1")

        x = jnp.dot(x, self.projection)

        x = jnp.pi * x

        x = jnp.hstack([jnp.sin(x), jnp.cos(x)])

        x = rearrange(x, "... -> (...)")

        return x


class GaussianFourierFeatureEncoding(eqx.Module):
    projection: Array = eqx.static_field()
    sigma: Array
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()

    def __init__(
        self,
        in_dim,
        num_features: int = 10,
        sigma: float = 1.0,
        key=jrandom.PRNGKey(123),
    ):
        projection = gausisan_rff(
            in_dim=in_dim, num_features=num_features, sigma=1.0, key=key
        )

        self.in_dim = projection.shape[0]
        self.projection = projection
        self.out_dim = num_features * 2
        self.sigma = jnp.asarray(sigma)
        self.num_features = num_features

    def __call__(self, x: Array, *, key=None) -> Array:

        # x = repeat(x, "... -> ... 1")

        x = jnp.dot(self.projection, x)

        x = jnp.pi * x

        x = jnp.hstack([jnp.sin(x), jnp.cos(x)])

        x = rearrange(x, "... -> (...)")

        x *= jnp.sqrt(self.sigma**2 / self.num_features)

        return x


class ArcCosineFourierFeatureEncoding(eqx.Module):
    projection: Array = eqx.static_field()
    sigma: Array
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()

    def __init__(
        self,
        in_dim,
        num_features: int = 10,
        sigma: float = 1.0,
        key=jrandom.PRNGKey(123),
    ):
        projection = gausisan_rff(
            in_dim=in_dim, num_features=num_features, sigma=1.0, key=key
        )

        self.in_dim = projection.shape[0]
        self.projection = projection
        self.out_dim = num_features * 2
        self.sigma = jnp.asarray(sigma)
        self.num_features = num_features

    def __call__(self, x: Array, *, key=None) -> Array:

        # x = repeat(x, "... -> ... 1")

        x = jnp.dot(self.projection, x)

        x = jnp.pi * x

        x = jnp.where(x > 0, x, 0.0)

        x = rearrange(x, "... -> (...)")

        x *= jnp.sqrt(self.sigma**2 / self.num_features)

        return x


def sincos_freq(num_features):
    return 2.0 ** jnp.arange(num_features)


def gausisan_rff(
    in_dim: int, num_features: int, sigma: float = 1.0, key=jrandom.PRNGKey(123)
):

    return sigma * jrandom.normal(key=key, shape=(num_features, in_dim))
