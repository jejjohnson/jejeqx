import typing as tp
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array


class VariationalCost(eqx.Module):
    prior: tp.Callable = eqx.static_field()
    obs_op: tp.Callable = eqx.static_field()
    prior_weight: Array = eqx.static_field()
    obs_op_weight: Array = eqx.static_field()
    background_weight: Array = eqx.static_field()

    def __init__(
        self,
        prior: tp.Callable,
        obs_op: tp.Callable,
        prior_weight: float = 0.8,
        obs_op_weight: float = 0.1,
        background_weight: float = 0.1,
    ):
        self.prior = prior
        self.obs_op = obs_op
        self.prior_weight = jnp.asarray(prior_weight)
        self.obs_op_weight = jnp.asarray(obs_op_weight)
        self.background_weight = jnp.asarray(background_weight)

    def loss(
        self,
        x,
        ts,
        y,
        mask: tp.Optional[Array] = None,
        xb: tp.Optional[Array] = None,
        x_gt: tp.Optional[Array] = None,
        return_loss: bool = False,
    ):
        if x_gt is None:
            x_gt = x

        if xb is None:
            xb = x[0]

        # prior loss
        prior_loss = self.prior.loss(x, ts, x_gt)

        # observation loss
        obs_loss = self.obs_op.loss(x, y, mask)

        # background loss
        background_loss = jnp.sum((x[0] - xb) ** 2)

        # compute variational loss
        var_loss = self.prior_weight * prior_loss
        var_loss += self.obs_op_weight * obs_loss
        var_loss += self.background_weight * background_loss

        # save other costs for auxillary outputs
        if return_loss:
            return var_loss, dict(
                var_loss=var_loss, prior=prior_loss, obs=obs_loss, bg=var_loss
            )
        else:
            return var_loss
