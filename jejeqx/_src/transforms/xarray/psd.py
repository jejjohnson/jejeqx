from typing import List
import xrft
import xarray as xr
from functools import reduce


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


def psd_average_freq(da: xr.DataArray, drop: bool = True) -> xr.DataArray:

    # get all frequency-domain dimensions
    freq_dims = [idim for idim in list(da.dims) if "freq" in idim]

    # get all real-domain dimensions
    dim = [idim for idim in list(da.dims) if "freq" not in idim]

    # create condition for frequency-domain
    if len(freq_dims) > 1:
        cond = reduce(lambda x, y: (da[x] > 0.0) & (da[y] > 0.0), freq_dims)
    else:
        cond = da[freq_dims[0]] > 0.0

    # take mean of real dimensions
    return da.mean(dim=dim).where(cond, drop=bool)
