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