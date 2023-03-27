from typing import NamedTuple
import equinox as eqx
import optax
import dataclasses
from jaxtyping import PyTree


class TrainState(eqx.Module):
    params: eqx.Module
    opt_state: optax.OptState
    tx: optax.GradientTransformation = eqx.static_field()

    def __init__(self, params: eqx.Module, tx: optax.GradientTransformation):
        self.params = params
        self.tx = tx
        self.opt_state = tx.init(params)

    @staticmethod
    def update_state(state: PyTree, grads: PyTree) -> PyTree:
        # apply gradients
        updates, opt_state = state.tx.update(
            grads, state.opt_state, params=state.params
        )

        params = optax.apply_updates(state.params, updates)
        state = eqx.tree_at(lambda x: x.params, state, params)
        state = eqx.tree_at(lambda x: x.opt_state, state, opt_state)

        return state
