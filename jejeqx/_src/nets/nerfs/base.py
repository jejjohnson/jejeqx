from typing import Optional
import equinox as eqx
from equinox import static_field
from jaxtyping import Array
import jax
import jax.random as jrandom
import jax.numpy as jnp

PRNGKey = jax.random.PRNGKey


class NerF(eqx.Module):
    basis_net: eqx.Module
    network: eqx.Module

    def __init__(self, basis_net, network):
        self.basis_net = basis_net
        self.network = network

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:

        x = self.basis_net(x, key=key)

        x = self.network(x, key=key)

        return x


class LatentNerF(eqx.Module):
    mod_basis_net: eqx.Module
    network: eqx.Module
    latent: Array

    def __init__(self, mod_basis_net, network, latent_dim, *, key):
        self.mod_basis_net = mod_basis_net
        self.network = network
        self.latent = jrandom.normal(key=key, shape=(latent_dim,))

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:

        x = self.mod_basis_net(x, self.latent, key=key)

        x = self.network(x, key=key)

        return x


class ShapeParamNerF(eqx.Module):
    mod_shape_net: eqx.Module
    param_net: eqx.Module
    network: eqx.Module

    def __init__(self, mod_shape_net, param_net, network):
        self.mod_shape_net = mod_shape_net
        self.param_net = param_net
        self.network = network

    def __call__(self, x: Array, mu: Array, *, key: Optional[PRNGKey] = None):

        z = self.param_net(mu)

        x = self.mod_shape_net(x, z)

        x = self.network(x)

        return x


class SpatioTempNerF(eqx.Module):
    basis_net: eqx.Module
    network: eqx.Module
    spatial_encoder: eqx.Module
    temporal_encoder: eqx.Module

    def __init__(
        self,
        basis_net: eqx.Module,
        network: eqx.Module = eqx.nn.Identity(),
        spatial_encoder: eqx.Module = eqx.nn.Identity(),
        temporal_encoder: eqx.Module = eqx.nn.Identity(),
        data_encoder: eqx.Module = eqx.nn.Identity(),
        *,
        key: jrandom.PRNGKey = jrandom.PRNGKey(123)
    ):
        self.basis_net = basis_net
        self.network = network
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder

    def __call__(self, x: Array, t: Array, *, key: Optional[PRNGKey] = None):

        x = self.spatial_encoder(x)

        t = self.temporal_encoder(t)

        x = jnp.hstack([x, t])

        x = self.basis_net(x)

        x = self.network(x)

        x = self.network(x, key=key)

        return x


class SpatioTempParamNerF(ShapeParamNerF):
    time_encoder: eqx.Module

    def __init__(self, mod_shape_net, param_net, network, time_encoder, *, key):
        super().__init__(
            mod_shape_net=mod_shape_net, param_net=param_net, network=network, key=key
        )
        self.time_encoder = time_encoder

    def __call__(self, x: Array, t: Array, mu: Array, *, key: Optional[PRNGKey] = None):

        t = self.time_encoder(t)

        mu = jnp.hstack([t, mu])

        return super().__call__(x=x, mu=mu, key=key)
