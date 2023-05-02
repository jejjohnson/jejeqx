import xarray as xr

def transform_360_to_180(ds: xr.Dataset) -> xr.Dataset:
    """This converts the coordinates that are bounded from
    [-180, 180] to coordinates bounded by [0, 360]

    Args:
        coord (np.ndarray): the input array of coordinates

    Returns:
        coord (np.ndarray): the output array of coordinates
    """
    ds["lon"] = (ds["lon"] + 180) % 360 - 180
    return ds


def transform_180_to_360(coord: xr.Dataset) -> xr.Dataset:
    """This converts the coordinates that are bounded from
    [0, 360] to coordinates bounded by [-180, 180]

    Args:
        coord (np.ndarray): the input array of coordinates

    Returns:
        coord (np.ndarray): the output array of coordinates
    """
    ds["lon"] = ds["lon"] % 360
    return ds