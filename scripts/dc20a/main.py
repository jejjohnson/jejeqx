import autoroot
import hydra
from loguru import logger
import jax
import jax.numpy as jnp

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    logger.info("Starting!")
    
    logger.info("Initializing datamodule...")
    dm = hydra.utils.instantiate(cfg.data)
    
    dm.setup()
    
    init = dm.ds_train[:32]
    x_init, t_init, y_init = init["spatial"], init["temporal"], init["data"]
    print(x_init.shape, t_init.shape)
    
    print(cfg.model)
    
    model = hydra.utils.instantiate(cfg.model)
    
    # check output of models
    out = jax.vmap(model)(jnp.hstack([x_init,x_init]))
    assert out.shape == y_init.shape

if __name__ == "__main__":
    main()