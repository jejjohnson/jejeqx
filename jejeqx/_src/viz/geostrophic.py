import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)




def plot_colormap(da: xr.DataArray, **kwargs):

    fig, ax = plt.subplots()
    
    da.plot.pcolormesh(ax=ax, **kwargs)
    
    plt.tight_layout()
    
    plt.show()
    return fig, ax


def plot_kinetic(da: xr.DataArray, **kwargs):

    fig, ax = plt.subplots()
    
    da.plot.pcolormesh(ax=ax, cmap="viridis", **kwargs)
    
    plt.tight_layout()
    
    plt.show()
    return fig, ax



def plot_vorticity(da: xr.DataArray, **kwargs):
    
    fig, ax = plt.subplots()
    
    da.plot.pcolormesh(ax=ax, cmap="RdBu_r", **kwargs)
    
    plt.tight_layout()
    
    plt.show()
    return fig, ax