import typing as tp
import equinox as eqx
import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array
from jejeqx._src.fourdvar.utils import time_patches

Solver = dfx.AbstractSolver
StepSize = dfx.AbstractAdaptiveStepSizeController


class DynamicalPrior(eqx.Module):
    params: eqx.Module = eqx.static_field()
    model: tp.Callable
    solver: Solver
    stepsize: StepSize

    def __init__(
        self,
        params: PyTree,
        model: tp.Callable,
        solver: Solver = dfx.Tsit5(),
        stepsize: StepSize = dfx.PIDController(rtol=1e-5, atol=1e-5),
    ):
        self.params = params
        self.model = model
        self.solver = solver
        self.stepsize = stepsize

    def init_state(self, x: PyTree) -> PyTree:
        raise NotImplementedError()

    def __call__(
        self,
        x: Array,
        ts: Array,
        dt: tp.Optional[float] = None,
        params: tp.Optional[PyTree] = None,
    ) -> PyTree:
        t0 = ts[0]
        t1 = ts[-1]

        # time step
        if dt is None:
            dt = ts[1] - ts[0]

        # initialize state
        state = self.init_state(x=x)

        saveat = dfx.SaveAt(ts=jnp.asarray([t1]))

        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.model),
            solver=self.solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=state,
            saveat=saveat,
            args=params if params is not None else self.params,
            stepsize_controller=self.stepsize,
        )

        return sol.ys

    def loss(self, x: Array, ts: Array, x_gt: tp.Optional[Array] = None) -> Array:
        """Dynamical Loss Function for Prior operator

        Eqn:
            R(u;θ) = || u - ϕ(u;θ)||

        Args:
            x (Array): an array of states, [timesteps, variables]
            ts (Array): an array of times, [timesteps, ]
            x_gt (Array): an array of true states, [timesteps, variables]

        Returns:
            loss (Array): a loss scalar value for the dynamical cost function
        """
        if x_gt is None:
            x_gt = x
        # create t batches
        ts = time_patches(ts)

        # check sizes
        msg = f"Size Mismatch: \n{x.shape} | {ts.shape} | {x_gt.shape}"
        assert len(x) - 1 == len(ts) == len(x_gt) - 1, msg

        # dynamical one-step predictions
        x_pred = jax.vmap(self, in_axes=(0, 0), out_axes=(0))(x[:-1], ts)

        # return an array
        x_pred = x_pred.array

        # mean squared error
        loss = jnp.sum((x_pred - x_gt[1:]) ** 2)
        return loss
