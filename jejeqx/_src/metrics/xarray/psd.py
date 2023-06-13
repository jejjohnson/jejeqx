from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import warnings
from jejeqx._src.transforms.xarray.psd import (
    psd_spacetime,
    psd_isotropic,
    psd_average_freq,
)


def psd_isotropic_error(
    da: xr.DataArray,
    da_ref: xr.DataArray,
    dims: List[str],
) -> xr.DataArray:
    return psd_average_freq(psd_isotropic(da_ref - da, dims))


def psd_spacetime_error(
    da: xr.DataArray,
    da_ref: xr.DataArray,
    dims: List[str],
) -> xr.DataArray:
    return psd_average_freq(psd_spacetime(da_ref - da, dims))


def psd_isotropic_score(
    da: xr.DataArray, da_ref: xr.DataArray, dims: List[str]
) -> xr.DataArray:

    # error
    score = psd_isotropic_error(da=da, da_ref=da_ref, dims=dims)

    # reference signal
    psd_ref = psd_average_freq(psd_isotropic(da_ref, dims))

    # normalized score
    score = 1.0 - (score / psd_ref)

    return score


def psd_spacetime_score(
    da: xr.DataArray, da_ref: xr.DataArray, dims: List[str]
) -> xr.DataArray:

    # error
    score = psd_spacetime_error(da=da, da_ref=da_ref, dims=dims)

    # reference signal
    psd_ref = psd_average_freq(psd_spacetime(da_ref, dims))

    # normalized score
    score = 1.0 - (score / psd_ref)

    return score


def psd_isotropic_resolved_scale(
    score: xr.DataArray, level: float = 0.5
) -> xr.DataArray:
    score.attrs["resolved_scale_space"] = find_intercept_1D(
        x=score.values, y=1.0 / score.freq_r.values, level=level
    )

    return score


def psd_spacetime_resolved_scale(
    score: xr.DataArray, levels: Union[float, List[float]] = 0.5
) -> xr.DataArray:

    lon_rs, time_rs = find_intercept_2D(
        x=1.0 / score.freq_lon.values,
        y=1.0 / score.freq_time.values,
        z=score.values,
        levels=levels,
    )
    score.attrs["resolved_scale_space"] = lon_rs
    score.attrs["resolved_scale_time"] = time_rs

    return score


def find_intercept_2D(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, levels: Union[float, List[float]] = 0.5
) -> Tuple[float, float]:

    x_shape = x.shape[0]
    y_shape = y.shape[0]
    z = z.reshape((y_shape, x_shape))

    if not isinstance(levels, list):
        levels = [levels]

    cs = plt.contour(x, y, z, levels=levels)
    try:
        x_level, y_level = cs.collections[0].get_paths()[0].vertices.T
    except IndexError:
        x_level, y_level = np.inf, np.inf
    plt.close()

    return np.min(x_level), np.min(y_level)


def find_intercept_1D(x, y, level=0.5):

    f = interp1d(x, y)

    try:
        ynew = f(level)
    except ValueError:
        text = f"The interpolated value is outside the range. {level}|{x.min():.2f}-{x.max():.2f}"
        warnings.warn(text)
        if level < x.min():
            ynew = f(x.min())
        else:
            ynew = f(x.max())

    return ynew
