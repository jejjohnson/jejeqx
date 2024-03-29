from metpy.calc import lat_lon_grid_deltas
import numpy as np
import xarray as xr
from metpy.units import units


def latlon_deg2m(ds: xr.Dataset, mean: bool = True):

    ds = ds.copy()

    out = lat_lon_grid_deltas(ds.lon, ds.lat)

    dx = out[0][:, 0]
    dy = out[1][0, :]

    num_dx = len(dx)
    num_dy = len(dy)

    lon_attrs = ds["lon"].attrs
    lat_attrs = ds["lat"].attrs

    if mean:
        ds["lon"] = np.arange(0, num_dx) * np.mean(dx)
        ds["lat"] = np.arange(0, num_dy) * np.mean(dy)
    else:
        ds["lon"] = np.cumsum(dx)
        ds["lat"] = np.cumsum(dy)

    ds["lon"] = ds["lon"].values * units.meters
    ds["lat"] = ds["lat"].values * units.meters

    ds["lon"].attrs = lon_attrs
    ds["lat"].attrs = lat_attrs

    ds["lon"].attrs["units"] = "m"
    ds["lat"].attrs["units"] = "m"

    return ds


def time_rescale(ds: xr.Dataset, freq_dt: int = 1, freq_unit: str = "D"):

    ds = ds.copy()

    t0 = ds["time"].min()

    ds["time"] = (ds["time"] - t0) / np.timedelta64(freq_dt, freq_unit)

    return ds
