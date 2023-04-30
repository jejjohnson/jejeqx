import equinox as eqx
from equinox import static_field
import jax.numoy as jnp


class Deg2Rad(eqx.Module):
    def __call__(self, x, inverse: bool=False):
        
        if inverse:
            return jnp.rad2deg(x)
        else:
            return jnp.deg2rad(x)
    

class Rad2Deg(eqx.Module):
    def __call__(self, x, inverse: bool=False):
        
        if inverse:
            return super().__call__(x, inverse=False)
        else:
            return super().__call__(x, inverse=True)

        
class Constant(eqx.Module):
    constant: Array = static_field()
    
    def __init__(self, constant: float):
        self.constant = jnp.asarray(constant)
        
    def __call__(self, x, inverse: bool=False):
        if inverse:
            return self.constant * x
        else:
            return x / self.constant
    

class Identity(eqx.Module):
    def __call__(self, x):
        return x


class MinMaxScaler(eqx.Module):
    input_min: Array = static_field() 
    input_max: Array = static_field()
    output_min: Array = static_field()
    output_max: Array = static_field()
        