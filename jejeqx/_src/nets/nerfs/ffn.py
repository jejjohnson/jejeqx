from jaxtyping import Array
from einops import repeat, rearrange
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


class RFFARD(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    
    def __init__(
        self, 
        in_dim, 
        num_features: int=10, 
        variance: float=0.1, 
        length_scale: float=0.01,
        *,
        key=jrandom.PRNGKey(123)
    ):
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))
                    
        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        self.log_variance = jnp.asarray(variance)
        self.log_length_scale = jnp.asarray(length_scale)
        self.num_features = num_features
        
    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)
    
    @property
    def variance(self):
        return jnp.exp(self.log_variance)
        
    def __call__(self, x: Array, *, key=None) -> Array:
        
                
        x /= self.length_scale
        
        x = jnp.dot(self.omega, x)
                
        x = jnp.hstack([jnp.sin(x), jnp.cos(x)])
        
        x = rearrange(x, "... -> (...)")
        
        x *= jnp.sqrt(self.variance**2/self.num_features)
                
        return x

    
class RFFARDCosine(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    
    def __init__(
        self, 
        in_dim, 
        num_features: int=10, 
        variance: float=0.1, 
        length_scale: float=0.01,
        *,
        key=jrandom.PRNGKey(123)
    ):
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))
                    
        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        self.log_variance = jnp.asarray(variance)
        self.log_length_scale = jnp.asarray(length_scale)
        self.num_features = num_features
        
    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)
    
    @property
    def variance(self):
        return jnp.exp(self.log_variance)
        
        
    def __call__(self, x: Array, *, key=None) -> Array:
        
                
        x /= self.length_scale
        
        x = jnp.dot(self.omega, x) + self.bias
        
        x = jnp.cos(x)
        
        x = rearrange(x, "... -> (...)")
        
        x *= jnp.sqrt(self.variance**2/self.num_features)
                
        return x
    

class RFFArcCosine(eqx.Module):
    log_variance: Array
    log_length_scale: Array
    omega: Array = eqx.static_field()
    bias: Array = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    num_features: int = eqx.static_field()
    
    def __init__(
        self, 
        in_dim, 
        num_features: int=10, 
        variance: float=0.1, 
        length_scale: float=0.1,
        *,
        key=jrandom.PRNGKey(123)
    ):
        okey, bkey = jrandom.uniform(key=key, shape=2)
        omega = jrandom.normal(key=key, shape=(num_features, in_dim))
        beta = jrandom.uniform(key=key, shape=(num_features,))
                    
                    
        self.in_dim = in_dim
        self.omega = omega
        self.out_dim = num_features * 2
        self.log_variance = jnp.asarray(sigma)
        self.num_features = num_features
        
    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale)
    
    @property
    def variance(self):
        return jnp.exp(self.log_variance)
        
    def __call__(self, x: Array, *, key=None) -> Array:
        
                
        x /= self.length_scale
        
        x = jnp.where(x > 0, x, 0.0)
        
        x = rearrange(x, "... -> (...)")
        
        x *= jnp.sqrt(self.variance**2/self.num_features)
                
        return x    

    
class RFFLayer(eqx.Module):
    rff_layer: RFFARD
    linear_layer: eqx.nn.Linear
    
    def __init__(
        self, 
        in_dim,
        out_dim,
        num_features: int=10, 
        variance: float=0.1, 
        length_scale: float=0.01,
        use_bias: bool=True,
        method: str="ard",
        *,
        key=jrandom.PRNGKey(123)
    ):
        if method == "ard_cos":
            self.rff_layer = RFFARDCosine(
                in_dim=in_dim, num_features=num_features,
                variance=variance, length_scale=length_scale,
                key=key
            )
        elif method == "ard":
            self.rff_layer = RFFARD(
                in_dim=in_dim, num_features=num_features,
                variance=variance, length_scale=length_scale,
                key=key
            )
        elif method == "arcosine":
            self.rff_layer = RFFArcCosine(
                in_dim=in_dim, num_features=num_features,
                variance=variance, length_scale=length_scale,
                key=key
            )
        else:
            raise ValueError(f"Unrecognized method: {method}")
        
        self.linear_layer = eqx.nn.Linear(
            in_features=self.rff_layer.out_dim,
            out_features=out_dim,
            use_bias=use_bias,
            key=key
        )
        
    def __call__(self, x: Array, *, key=None) -> Array:
        
        x = self.rff_layer(x, key=key)
        x = self.linear_layer(x, key=key)
                
        return x    