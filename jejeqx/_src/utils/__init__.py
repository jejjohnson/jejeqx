from omegaconf import OmegaConf


def dataclass_to_dict(params, **kwargs):
    
    params = OmegaConf.structured(params)
    
    params = OmegaConf.to_container(params, **kwargs)
    
    return params