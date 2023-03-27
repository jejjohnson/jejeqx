import pytest
import typing
import equinox as eqx
import jax.random as jrandom
from eqx_nerf._src.siren import sine_activation, Sine, Siren, SirenNet
import random
import numpy as np
import jax.numpy as jnp

typing.TESTING = True  # pyright: ignore


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        # return jrandom.PRNGKey(random.randint(0, 2**31 - 1))
        return jrandom.PRNGKey(random.randint(-1, 1))

    return _getkey


# TODO: Fix Sine Activation Test!
# def test_sine_activation(getkey):
#     x = jrandom.normal(getkey(), (3,))

#     w0 = jnp.asarray([0.1])
#     out = sine_activation(x, w0)
#     out_ = jnp.sin(w0 * x)

#     assert jnp.allclose(out, out_)


def test_sine_activation_layer(getkey):
    x = jrandom.normal(getkey(), (3,))

    w0 = 0.1
    out = Sine(w0=w0)(x)
    out_ = jnp.sin(w0 * x)

    assert jnp.allclose(out, out_)


def test_siren_layer(getkey):
    # Positional arguments
    siren = Siren(3, 4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert siren(x).shape == (4,)

    # Some keyword arguments
    linear = Siren(3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = Siren(in_features=3, out_features=4, key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    linear = Siren("scalar", 2, key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = Siren(2, "scalar", key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()


def test_siren_net(getkey):
    w0_init = 30.0
    w0 = 1.0
    c = 6.0

    siren = SirenNet(2, 3, 8, 2, w0_init, w0, c, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert siren(x).shape == (3,)
    assert siren.layers[0].w0 == w0_init
    assert siren.layers[0].c == c
    assert siren.layers[1].w0 == w0
    assert siren.layers[1].c == c

    siren = SirenNet(
        in_size=2,
        out_size=3,
        width_size=8,
        depth=2,
        w0_initial=30.0,
        w0=1.0,
        c=6.0,
        key=getkey(),
    )

    x = jrandom.normal(getkey(), (2,))
    assert siren(x).shape == (3,)

    siren = SirenNet("scalar", 2, 2, 2, 30.0, 1.0, 6.0, key=getkey())
    x = jrandom.normal(getkey(), ())
    assert siren(x).shape == (2,)

    siren = SirenNet(2, "scalar", 2, 2, 30.0, 1.0, 6.0, key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert siren(x).shape == ()
