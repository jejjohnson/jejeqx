from typing import List
import xrft
import xarray as xr

def psd_spacetime(da: xr.DataArray, dims: List[str], **kwargs) -> xr.DataArray:
    
    # compute PSD err and PSD signal
    psd_signal = xrft.power_spectrum(
        da,
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )


    return psd_signal


def psd_isotropic(da: xr.DataArray, dims: List[str], **kwargs) -> xr.DataArray:
    
    # compute PSD err and PSD signal
    psd_signal = xrft.isotropic_power_spectrum(
        da,
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )


    return psd_signal



def psd_average_freq(da, drop: bool=True):
    
    freq_dims = [idim for idim in list(da.dims) if "freq" in idim]
    dim = [idim for idim in list(da.dims) if "freq" not in idim]
    
    cond = [(da[idim]>0.0) for idim in freq_dims]
    
    return da.mean(dim=dim).where(*cond, drop=bool)