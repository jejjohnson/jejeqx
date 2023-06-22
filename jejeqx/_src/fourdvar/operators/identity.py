import equinox as eqx


class Identity(eqx.Module):
    def __call__(self, x):
        return x
