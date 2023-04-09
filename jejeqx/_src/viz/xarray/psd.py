import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker


def plot_psd_isotropic_wavenumber(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs):
    
    if scale == "km":
        scale = 1e3
        xlabel="Wavenumber [cycles/km]"
    else:
        scale = 1.0
        xlabel="Wavenumber [cycles/m]"
        
    if units is None:
        ylabel = "PSD"
    else:
        ylabel = f"PSD [{units}]"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    else:
        fig = plt.gcf()

    ax.plot(da.freq_r * scale, da, **kwargs)

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel=ylabel,
    )

    ax.legend()
    ax.grid(which="both", alpha=0.5)

    return fig, ax


def plot_psd_isotropic_wavelength(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs
):
    
    
    if scale == "km":
        xlabel="Wavenumber [km]"
    else:
        xlabel="Wavenumber [m]"
        
    if units is None:
        ylabel = "PSD"
    else:
        ylabel = f"PSD [{units}]"

    fig, ax = plot_psd_isotropic_wavenumber(
        da, 
        ax=ax, 
        scale=scale, 
        units=units,
        **kwargs
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel=ylabel,
    )

    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.invert_xaxis()

    return fig, ax


def plot_psd_isotropic(
    da: xr.DataArray, scale: str="km", units: str=None, ax=None, **kwargs
):
    
    if scale == "km":
        xlabel="Wavenumber [km]"
    else:
        xlabel="Wavenumber [m]"

    fig, ax = plot_psd_isotropic_wavenumber(
        da=da, ax=ax, scale=scale, units=units, **kwargs
    )

    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.set(xlabel=xlabel)

    return fig, ax, secax


def plot_psd_spacetime_wavenumber(
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

    locator = ticker.LogLocator()
    norm = colors.LogNorm()
    

    pts = ax.contourf(
        1/(da.freq_lon*space_scale),
        1/da.freq_time, 
        da.transpose("freq_time", "freq_lon"), 
        norm=norm, 
        locator=locator, 
        cmap="RdYlGn", 
        extend="both"
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel=xlabel,
        ylabel="Frequency [cycles/days]",
    )
    # colorbar
    fmt = ticker.LogFormatterMathtext(base=10)
    cbar = plt.colorbar(
        pts,
        ax=ax,
        pad=0.02,
        format=fmt,
    )
    cbar.ax.set_ylabel(cbar_label)
    
    ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    return fig, ax, cbar


def plot_psd_spacetime_wavelength(da, space_scale="km", psd_units=None, ax=None):
    
    if space_scale == "km":
        xlabel = "Wavelength [km]"
    elif space_scale == "m":
        xlabel = "Wavelength [m]"
    elif space_scale is None:
        xlabel = "Wavelength k"
    else:
        raise ValueError(f"Unrecognized scale: {space_scale}")
        
    fig, ax, cbar = plot_psd_spacetime_wavenumber(
        da, space_scale=space_scale, psd_units=psd_units, ax=ax)
    ax.set(yscale="log", xscale="log", xlabel=xlabel, ylabel="Period [days]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_major_formatter("{x:.0f}")

    return fig, ax, cbar