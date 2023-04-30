from typing import Optional, Union
import numpy as np
import xarray as xr
import pandas as pd

UNITS = {
    "ns":"nanoseconds",
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "seconds", 
    "m": "minutes", 
    "h": "hours", 
    "D": "days", 
    "W":"weeks",
    "M": "months",
    "Y": "years"
}


def decode_cf_time(da, units='seconds since 2012-10-01'):
    da = da.copy()
    if units is not None:
        da["time"] = da.time.assign_attrs(units=units)
    return xr.decode_cf(da)


def validate_time(da):
    da = da.copy()
    da["time"] = pd.to_datetime(da.time)
    return da


def time_rescale(
    ds: xr.Dataset,
    freq_dt: int=1,
    freq_unit: str="seconds",
    t0: Optional[Union[str, np.datetime64]]=None
) -> xr.Dataset:
    """Rescales time dimensions of np.datetim64 to an output frequency.
    
    t' = (t - t_0) / dt
    
    Args:
        ds (xr.Dataset): the xr.Dataset with a time dimensions
        freq_dt (int): the frequency of the temporal coordinate
        freq_unit (str): the unit for the time frequency parameter
        t0 (datetime64, str): the starting point. Optional. If none, assumes the 
            minimum value of the time coordinate
    
    Returns:
        ds (xr.Dataset): the xr.Dataset with the rescaled time dimensions in the 
            freq_unit.
    """
    
    ds = ds.copy()
    
    
    if t0 is None:
        t0 = ds["time"].min()
    
    if isinstance(t0, str):
        t0 = np.datetime64(t0)
    
    ds["time"] = ((ds["time"] - t0 ) / pd.to_timedelta(freq_dt, freq_unit)).astype(np.float32)
    
    return ds