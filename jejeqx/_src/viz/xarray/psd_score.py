import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
from jejeqx._src.viz.xarray.psd import plot_psd_isotropic


def plot_psd_isotropic_score(
    da: xr.DataArray, 
    scale="km", 
    ax=None, 
    color: str="k", 
    name: str="model", 
    **kwargs
):
    
    if scale == "km":
        factor = 1e3
    elif scale == "m":
        factor = 1.0
        rfactor = 1
        
    else:
        raise ValueError(f"Unrecognized scale")
    
    fig, ax, secax = plot_psd_isotropic(
        da=da, scale=scale, ax=ax, **kwargs
    )
    
    ax.set(ylabel="PSD Score", yscale="linear")
    ax.set_ylim((0,1.0))
    ax.set_xlim((
        np.ma.min(np.ma.masked_invalid(da.freq_r.values * factor)),
        np.ma.max(np.ma.masked_invalid(da.freq_r.values * factor)),
    ))
    
    resolved_scale = factor / da.attrs["resolved_scale"]
    
    ax.vlines(
        x=resolved_scale, ymin=0, ymax=0.5, color=color, linewidth=2, 
        linestyle="--",
    )
    ax.hlines(
        y=0.5,
        xmin=np.ma.min(np.ma.masked_invalid(da.freq_r.values * factor)),
        xmax=resolved_scale, color=color,
        linewidth=2, linestyle="--",
    )
    
    ax.set_aspect("equal", "box")
    
    label = f"{name}: {1/resolved_scale:.0f} {scale} "
    ax.scatter(
        resolved_scale, 0.5, 
        color=color, marker=".", linewidth=5, label=label,
        zorder=3
    )
    
    return fig, ax, secax


def plot_psd_spacetime_wavenumber_score(
    da: xr.DataArray, 
    space_scale: str=None,
    psd_units: str=None,
    ax=None
):
    
    if space_scale == "km":
        space_scale = 1e3
        xlabel = "Wavenumber [cycles/km]"
    elif space_scale == "m":
        space_scale = 1.0
        xlabel = "Wavenumber [cycles/m]"
    elif space_scale is None:
        space_scale = 1.0
        xlabel = "Wavenumber [k]"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")
        
    if psd_units is None:
        cbar_label = "PSD"
    else:
        cbar_label = f"PSD [{psd_units}]"
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    
    pts = ax.contourf(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        extend="both",
        cmap="RdBu", 
        levels=np.arange(0, 1.1, 0.1)
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel="Frequency [cycles/days]",
    )
    # colorbar

    cbar = fig.colorbar(
        pts,
        pad=0.02,
    )
    cbar.ax.set_ylabel(cbar_label)

    plt.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    pts_middle = ax.contour(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        levels=[0.5], 
        linewidths=2, 
        colors="k"
    )

    cbar.add_lines(pts_middle)

    return fig, ax, cbar


def plot_psd_spacetime_score_wavelength(
    da, space_scale="km", psd_units=None, ax=None
):
    
    if space_scale == "km":
        xlabel = "Wavelength [km]"
    elif space_scale == "m":
        xlabel = "Wavelength [m]"
    elif space_scale is None:
        xlabel = "Wavelength k"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")

    fig, ax, cbar = plot_psd_spacetime_wavenumber_score(
        da, space_scale=space_scale, psd_units=psd_units, ax=ax)

    ax.set(yscale="log", xscale="log", xlabel=xlabel, ylabel="Period [days]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_major_formatter("{x:.0f}")

    return fig, ax, cbar