import xarray as xr
import numpy as np
from metpy.calc import lat_lon_grid_deltas
from metpy.units import units
import pint_xarray

def spherical_to_cartesian_3d(lon, lat, radius: float=6371.010):
    
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    
    return x, y, z


def cartesian_to_spherical_3d(x, y, z):
    
    radius = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arcsin( z / radius)
    
    return lon, lat, radius


def validate_lon(da):
    try:
        da = da.rename({"longitude": "lon"})
        da = da.assign_coord({"lon": da.lon})
    except:
        pass
    
    da["lon"] = (da.lon + 180) % 360 - 180
    da["lon"] = da.lon.assign_attrs(
        **{**dict(units="degrees_east",
        standard_name="longitude",
        long_name="Longitude",),
        **da.lon.attrs}
    )
    return da


def validate_lat(da):
    try:
        da = da.rename({"latitude": "lat"})
        da = da.assign_coord({"lat": da.lat})
    except:
        pass
    
    da["lat"] = (da.lat + 90) % 180 - 90
    da["lat"] = da.lat.assign_attrs(
        **{**dict(units="degrees_west",
        standard_name="latitude",
        long_name="Latitude",),
        **da.lat.attrs}
    )
    return da


def latlon_deg2m(ds: xr.Dataset, mean: bool=True) -> xr.Dataset:
    """Converts the lat/lon coordinates from degrees to meters
    
    Args:
        ds (xr.Dataset): the dataset with the lat/lon variables
        mean (bool): the whether to use the mean dx/dy for each
            lat/lon coordinate (default=True)
    
    Returns:
        ds (xr.Dataset): the xr.Dataset with the normalized lat/lon coords
    """
    ds = ds.copy()
    
    
    lon_attrs = ds["lon"].attrs
    lat_attrs = ds["lat"].attrs
    
    out = lat_lon_grid_deltas(ds.lon * units.degree, ds.lat * units.degree)
        
    dx = out[0][:, 0]
    dy = out[1][0, :]
    
    num_dx = len(dx)
    num_dy = len(dy)
    
    
    if mean:
        lat = np.arange(0, num_dx) * np.mean(dx)
        lon = np.arange(0, num_dy) * np.mean(dy)
    else:
        dx0, dy0 = dx[0], dy[0]
        lat = np.cumsum(dx) - dx0
        lon = np.cumsum(dy) - dy0
    
    lon_attrs.pop("units", None)
    lat_attrs.pop("units", None)
        
    ds["lon"] = lon
    ds["lat"] = lat
    ds["lon"].attrs = lon_attrs
    ds["lat"].attrs = lat_attrs
    
    # ds = ds.pint.quantify(
    #     {"lon": "meter", "lat": "meter"}
    # )
    return ds