import autoroot
from typing import List
import matplotlib.pyplot as plt
import xarray as xr
import cmocean as cmo
import numpy as np
import jejeqx._src.transforms.xarray.geostrophic as geocalc
from jejeqx._src.transforms.xarray.grid import latlon_deg2m, time_rescale
from jejeqx._src.transforms.xarray.psd import psd_spacetime, psd_isotropic, psd_average_freq
from jejeqx._src.viz.xarray.psd import plot_psd_isotropic
from jejeqx._src.viz.utils import get_cbar_label


def calculate_physical_quantities(da: xr.DataArray) -> xr.Dataset:
    
    # SSH
    ds = geocalc.get_ssh_dataset(da)
    
    # Stream Function
    ds = geocalc.calculate_streamfunction(ds, "ssh")
    
    # U,V Velocities
    ds = geocalc.calculate_velocities_sf(ds, "psi")
    
    # Kinetic Energy
    ds = geocalc.calculate_kinetic_energy(ds, ["u","v"])
    
    # Relative Vorticity
    ds = geocalc.calculate_relative_vorticity_uv(ds, ["u","v"], normalized=True)
    
    # Strain
    ds = geocalc.calculate_strain_magnitude(ds, ["u","v"], normalized=True)
    
    # Okubo-Weiss
    ds = geocalc.calculate_okubo_weiss(ds, ["u","v"], normalized=True)
    
    return ds


def calculate_isotropic_psd(ds, freq_dt=1, freq_unit="D"):
    
    ds = latlon_deg2m(ds, mean=True)
    ds = time_rescale(ds, freq_dt, freq_unit)
    
    # calculate isotropic PSDs
    ds_psd = xr.Dataset()
    ds_psd["ssh"] = psd_average_freq(psd_isotropic(ds.ssh, ["lat", "lon"]))
    ds_psd["u"] = psd_average_freq(psd_isotropic(ds.u, ["lat", "lon"]))
    ds_psd["v"] = psd_average_freq(psd_isotropic(ds.v, ["lat", "lon"]))
    ds_psd["ke"] = psd_average_freq(psd_isotropic(ds.ke, ["lat", "lon"]))
    ds_psd["vort_r"] = psd_average_freq(psd_isotropic(ds.vort_r, ["lat", "lon"]))
    ds_psd["strain"] = psd_average_freq(psd_isotropic(ds.strain, ["lat", "lon"]))
    ds_psd["ow"] = psd_average_freq(psd_isotropic(ds.ow, ["lat", "lon"]))

    return ds_psd


def plot_analysis_vars(ds: List[xr.Dataset], names: List[str]=None):
    
    ncols = len(ds)
    
    fig, ax = plt.subplots(nrows=7, ncols=ncols, figsize=(12, 20))
    
    # SSH
    vmin = np.min([ids.ssh.min() for ids in ds])
    vmax = np.max([ids.ssh.max() for ids in ds])
    for iax, ids in zip(ax[0], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ssh)}
        ids.ssh.plot.pcolormesh(
            ax=iax, cmap="viridis", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
    
    # U
    vmin = np.min([ids.u.min() for ids in ds])
    vmax = np.max([ids.u.max() for ids in ds])
    for iax, ids in zip(ax[1], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.u)}
        ids.u.plot.pcolormesh(
            ax=iax, cmap="gray", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
        )
    
    # v
    vmin = np.min([ids.v.min() for ids in ds])
    vmax = np.max([ids.v.max() for ids in ds])
    for iax, ids in zip(ax[2], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.v)}
        ids.v.plot.pcolormesh(
            ax=iax, cmap="gray", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    
    # Kinetic Energy
    vmin = np.min([ids.ke.min() for ids in ds])
    vmax = np.max([ids.ke.max() for ids in ds])
    for iax, ids in zip(ax[3], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ke)}
        ids.ke.plot.pcolormesh(
            ax=iax, cmap="YlGnBu_r", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    
    # Relative Vorticity
    vmin = np.min([ids.vort_r.min() for ids in ds])
    vmax = np.max([ids.vort_r.max() for ids in ds])
    for iax, ids in zip(ax[4], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.vort_r)}
        ids.vort_r.plot.pcolormesh(
            ax=iax, cmap="RdBu_r", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    
    
    # STRAIN
    vmin = np.min([ids.strain.min() for ids in ds])
    vmax = np.max([ids.strain.max() for ids in ds])
    for iax, ids in zip(ax[5], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.strain)}
        ids.strain.plot.pcolormesh(
            ax=iax, cmap=cmo.cm.speed, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        
    # Okubo-Weiss
    vmin = np.min([ids.ow.min() for ids in ds])
    vmax = np.max([ids.ow.max() for ids in ds])
    for iax, ids in zip(ax[6], ds):
        cbar_kwargs = {"label": get_cbar_label(ids.ow)}
        ids.ow.plot.contourf(
            ax=iax, cmap="cividis", vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    
    if names is not None:
        fig.suptitle(t=names)
        
    plt.tight_layout()
    return fig, ax



def plot_analysis_psd_iso(ds: List[xr.Dataset], names: List[str]):
    
    ncols = len(ds)
    
    fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(5, 25))
    
    
    # SSH
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "$m^{2}$/cycles/m"
        plot_psd_isotropic(ids.ssh, units=units, scale=scale, ax=ax[0], label=iname)

    
    # U
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "U-Velocity"
        plot_psd_isotropic(ids.u, units=units, scale=scale, ax=ax[1], label=iname)
    
    # v
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "V-Velocity"
        plot_psd_isotropic(ids.v, units=units, scale=scale, ax=ax[2], label=iname)
    
    # Kinetic Energy
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Kinetic Energy"
        plot_psd_isotropic(ids.ke, units=units, scale=scale, ax=ax[3], label=iname)

    # Relative Vorticity
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Relative Vorticity"
        plot_psd_isotropic(ids.vort_r, units=units, scale=scale, ax=ax[4], label=iname)
    
    
    # STRAIN
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Strain"
        plot_psd_isotropic(ids.strain, units=units, scale=scale, ax=ax[5], label=iname)
        
    # STRAIN
    for iname, ids in zip(names, ds):
        scale = "km"
        units = "Okubo-Weiss"
        plot_psd_isotropic(ids.ow, units=units, scale=scale, ax=ax[6], label=iname)
    
    
    
    plt.tight_layout()
    return fig, ax