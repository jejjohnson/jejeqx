import xarray as xr
import matplotlib.pyplot as plt


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