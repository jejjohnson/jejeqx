import typing as tp
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from jejeqx._src.fourdvar.varcost.base import VariationalCost

WeakVarCost = VariationalCost


class StrongVarCost(VariationalCost):
    def loss(
        self,
        x,
        ts,
        y,
        mask: tp.Optional[Array] = None,
        xb: tp.Optional[Array] = None,
        return_loss: bool = False,
    ):
        if xb is None:
            xb = x

        # prior loss
        x = self.prior(x, ts)

        x = x.array

        # observation loss
        obs_loss = self.obs_op.loss(x, y, mask)

        # background loss
        background_loss = jnp.sum((x[0] - xb) ** 2)

        # compute variational loss
        var_loss = self.obs_op_weight * obs_loss
        var_loss += self.background_weight * background_loss

        # save other costs for auxillary outputs
        if return_loss:
            return var_loss, dict(var_loss=var_loss, obs=obs_loss, bg=var_loss)
        else:
            return var_loss
