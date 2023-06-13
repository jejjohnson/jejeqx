import xarray as xr


def get_cbar_label(da: xr.DataArray) -> str:
    name = da.attrs["long_name"]
    units = da.attrs["units"]
    if units == "":
        label = f"{name}"
    else:
        label = f"{name} [{units}]"
    return label
