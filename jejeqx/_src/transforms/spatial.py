import xarray as xr
import numpy as np


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
