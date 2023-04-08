from typing import List
import metpy
import pint
import numpy as np
import xarray as xr
from metpy.constants import earth_gravity as GRAVITY



def get_ssh_dataset(da: xr.DataArray) -> xr.Dataset:
    
    da = da * pint.Unit('metres')
    
    da.name = "ssh"
    da.attrs["units"] = "m"
    da.attrs["long_name"] = "Sea Surface Height"
    da.attrs["standard_name"] = "sea_surface_height"
    
    try:
        da.time.attrs["long_name"] = "Time"
        da.time.attrs["standard_name"] = "time"
        da.time.attrs["units"] = ""
    except:
        pass
    
    da.lon.attrs["units"] = "°"
    da.lon.attrs["long_name"] = "Longitude"
    da.lon.attrs["standard_name"] = "longitude"
    
    da.lat.attrs["units"] = "°"
    da.lat.attrs["long_name"] = "Latitude"
    da.lat.attrs["standard_name"] = "latitude"
    
    return da.to_dataset()


def calculate_coriolis(lat: np.ndarray, mean: bool=True):
    
    f = metpy.calc.coriolis_parameter(latitude=np.deg2rad(lat))
    
    if mean:
        return f.mean()
    else:
        return f


def calculate_streamfunction(ds, variable: str="ssh", g: float=GRAVITY):
    
    f0 = calculate_coriolis(ds.lat.values, mean=True)
        
    psi = (g/f0) * ds[variable].data
    
    ds["psi"] = (("time", "lat", "lon"), psi)
    ds["psi"].attrs["units"] = f"{psi.u:~P}"
    ds["psi"].attrs["long_name"] = "Stream Function"
    ds["psi"].attrs["standard_name"] = "stream_function"
    
    return ds


def calculate_velocities_sf(ds, variable: str="psi"):
    
    dpsi_dx, dpsi_dy = metpy.calc.geospatial_gradient(
        f=ds[variable], latitude=ds.lat, longitude=ds.lon
    )
    
    ds["u"] = (("time", "lat", "lon"), - dpsi_dy)
    ds["u"].attrs["units"] = f"{dpsi_dy.u:~P}"
    ds["u"].attrs["long_name"] = "Zonal Velocity"
    ds["u"].attrs["standard_name"] = "zonal_velocity"
    
    ds["v"] = (("time", "lat", "lon"), dpsi_dx)
    ds["v"].attrs["units"] = f"{dpsi_dx.u:~P}"
    ds["v"].attrs["long_name"] = "Meridional Velocity"
    ds["v"].attrs["standard_name"] = "meridional_velocity"
    
    return ds


def calculate_relative_vorticity_sf(
    ds, variable: str="psi", normalized: bool=True):
    
    q = metpy.calc.geospatial_laplacian(
        f=ds[variable], latitude=ds.lat, longitude=ds.lon
    )
        
    ds["vort"] = (("time", "lat", "lon"), q.data)
    
    ds["vort"].attrs["long_name"] = "Relative Vorticity"
    ds["vort"].attrs["standard_name"] = "relative_vorticity"
    ds["vort"].attrs["units"] = f"{q.data.u:~P}"
    
    if normalized:
        ds = calculate_coriolis_normalized(ds, variable="vort")
    
    return ds

def calculate_relative_vorticity_uv(ds, variables: List[str]=["u", "v"], normalized: bool=True):
    
    _, du_dy = metpy.calc.geospatial_gradient(
        ds[variables[0]], latitude=ds.lat, longitude=ds.lon)
    dv_dx, _ = metpy.calc.geospatial_gradient(
        ds[variables[1]], latitude=ds.lat, longitude=ds.lon)
    
    zeta = dv_dx - du_dy
    
    ds["vort_r"] = (("time", "lat", "lon"), zeta.data)
    
    ds["vort_r"].attrs["long_name"] = "Relative Vorticity"
    ds["vort_r"].attrs["standard_name"] = "relative_vorticity"
    ds["vort_r"].attrs["units"] = f"{zeta.u:~P}"
    
    if normalized:
        ds = calculate_coriolis_normalized(ds, variable="vort_r")
    
    return ds

def calculate_absolute_vorticity_uv(ds, variables: List[str]=["u", "v"], normalized: bool=True):
    
    _, du_dy = metpy.calc.geospatial_gradient(
        ds[variables[0]], latitude=ds.lat, longitude=ds.lon)
    dv_dx, _ = metpy.calc.geospatial_gradient(
        ds[variables[1]], latitude=ds.lat, longitude=ds.lon)
    
    zeta = dv_dx + du_dy
    
    ds["vort_a"] = (("time", "lat", "lon"), zeta.data)
    
    ds["vort_a"].attrs["long_name"] = "Absolute Vorticity"
    ds["vort_a"].attrs["standard_name"] = "absolute_vorticity"
    ds["vort_a"].attrs["units"] = f"{zeta.u:~P}"
    
    if normalized:
        ds = calculate_coriolis_normalized(ds, variable="vort_a")
    
    return ds


def calculate_coriolis_normalized(ds, variable: str="q"):
    
    lname = ds[variable].attrs["long_name"]
    sname = ds[variable].attrs["standard_name"]
    
    f0 = calculate_coriolis(ds.lat.values, mean=True)
    
    var = ds[variable] / f0
    
    ds[variable] = (("time", "lat", "lon"), var.data)

    
    ds[variable].attrs["long_name"] = f"Normalized {lname}"
    ds[variable].attrs["standard_name"] = f"norm_{sname}"
    ds[variable].attrs["units"] = f"{var.data.u:~P}"
    
    return ds

def calculate_kinetic_energy(ds, variables: List[str]=["u", "v"]):
    
    ke = 0.5 * (ds[variables[0]]**2 + ds[variables[1]]**2)
    
    
    ds["ke"] = (("time", "lat", "lon"), ke.data)
    ds["ke"].attrs["units"] = f"{ke.data.u:~P}"
    ds["ke"].attrs["long_name"] = "Kinetic Energy"
    ds["ke"].attrs["standard_name"] = "kinetic_energy"
    
    return ds

def calculate_enstropy(ds, variable: str="q"):
    
    ens = 0.5 * (ds[variable]**2)
            
    ds["ens"] = (("time", "lat", "lon"), ens.data)
    ds["ens"].attrs["units"] = ""#f"{ds[variable].u**2:~P}"
    ds["ens"].attrs["long_name"] = "Enstropy"
    ds["ens"].attrs["standard_name"] = "enstropy"
    
    return ds

def calculate_strain_magnitude(ds, variables=["u", "v"], normalized: bool=True):
    
    du_dx, du_dy = metpy.calc.geospatial_gradient(
        ds[variables[0]], latitude=ds.lat, longitude=ds.lon)
    dv_dx, dv_dy = metpy.calc.geospatial_gradient(
        ds[variables[1]], latitude=ds.lat, longitude=ds.lon)
    
    strain_shear = dv_dx + du_dy
    strain_normal = du_dx - dv_dy
    
    strain_magnitude = np.hypot(strain_normal, strain_shear)
    
    ds["strain"] = (("time", "lat", "lon"), strain_magnitude)

    ds["strain"].attrs["long_name"] = "Strain"
    ds["strain"].attrs["standard_name"] = "strain"
    ds["strain"].attrs["units"] = f"{strain_magnitude.u:~P}"
    
    if normalized:
        ds = calculate_coriolis_normalized(ds, variable="strain")
    
    return ds
