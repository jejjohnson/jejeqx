import math
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from equinox import static_field
from einops import repeat
from jaxtyping import Array


class TimeIdentity(eqx.Module):
    out_features: int = static_field()
    def __init__(self, out_features: int, *, key: jrandom.PRNGKey):
        self.out_features = out_features
        
    def __call__(self, t: Array, *, key=None) -> Array:
        
        return repeat(t, "() -> D", D=self.out_features)
    

class TimeTanh(eqx.Module):
    scale: eqx.nn.Linear
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_bias: bool=True, 
        *, 
        key: jrandom.PRNGKey
    ):
        
        self.scale = eqx.nn.Linear(
            in_features=in_features, out_features=out_features, 
            use_bias=use_bias, key=key
        )
        
    
    def __call__(self, t: Array, *, key=None) -> Array:
        return jax.nn.tanh(self.scale(t))
    
    
class TimeLog(eqx.Module):
    scale: eqx.nn.Linear
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_bias: bool=True, 
        *, 
        key: jrandom.PRNGKey
    ):
        
        self.scale = eqx.nn.Linear(
            in_features=in_features, out_features=out_features, 
            use_bias=use_bias, key=key
        )
        
    
    def __call__(self, t: Array, *, key=None) -> Array:
        return jnp.log(jnp.exp(self.scale(t)) + 1)
    

class TimeFourier(eqx.Module):
    weight: Array
    shift: Array
    lmbd: float = static_field()
    bounded: bool = static_field()
    out_features: int = static_field()
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmbd: float=0.5,
        bounded: bool=False,
        *,
        key: jrandom.PRNGKey,
    ):
        wkey, skey = jrandom.split(key, 2)
        
        shift = jrandom.uniform(key=skey, shape=(out_features, in_features,))
        self.shift = - jnp.log(1- shift) / lmbd
        
        lim = 1 / math.sqrt(in_features)
        self.weight = jrandom.uniform(
            wkey, (out_features, in_features), minval=-lim, maxval=lim
        )
        
        self.bounded = bounded
        self.lmbd = lmbd
        self.out_features = out_features
        
    def get_scale(self):
        if self.bounded:
            return jax.nn.softmax(self.weight, -1) / 2
        else:
            return self.weight / self.out_features
        
    
    def __call__(self, t: Array, *, key=None) -> Array:
        
        scale = self.get_scale()
        
        t = scale * jnp.sin(self.shift * t)
        
        
        return t.squeeze()