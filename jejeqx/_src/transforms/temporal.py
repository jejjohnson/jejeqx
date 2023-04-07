import xarray as xr
import pandas as pd


def decode_cf_time(da, units='seconds since 2012-10-01'):
    da = da.copy()
    if units is not None:
        da["time"] = da.time.assign_attrs(units=units)
    return xr.decode_cf(da)


def validate_time(da):
    da = da.copy()
    da["time"] = pd.to_datetime(da.time)
    return da