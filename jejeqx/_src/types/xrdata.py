from typing import Literal
from dataclasses import dataclass
from xarray_dataclasses import Coordof, Dataof, Attr, Coord, Data, Name, AsDataArray, AsDataset, Dataof
import numpy as np
import pandas as pd


X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]
LON = Literal["lon"]
LAT = Literal["lat"]
HEIGHT = Literal["z"]
TIME = Literal["time"]


@dataclass
class Bounds:
    val_min: float
    val_max: float
    val_step: float
    name: str = ""


@dataclass
class Region:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    name: str = ""


@dataclass
class Period:
    t_min: str
    t_max: str
    dt_freq: int
    dt_unit: str
    name: str = ""
    
    @property
    def t_min_dt(self):
        return pd.to_datetime(self.t_min)
    
    @property
    def t_max_dt(self):
        return pd.to_datetime(self.t_max)
    
    @property
    def dt(self):
        return pd.to_timedelta(self.dt_freq, self.dt_unit)


@dataclass
class CoordinateAxis:
    data: Coord[Literal["x"], np.float32]

    @classmethod
    def init_from_limits(cls, x_min: float, x_max: float, dx: float, **kwargs):
        data = np.arange(x_min, x_max, dx)
        return cls(data=data, **kwargs)
    
    @property
    def ndim(self):
        return len(self.data)
    

class XAxis(CoordinateAxis):
    data: Data[X, np.float32]


class YAxis(CoordinateAxis):
    data: Data[Y, np.float32]

    @classmethod
    def init_from_limits(cls, y_min: float, y_max: float, dy: float, **kwargs):
        data = np.arange(y_min, y_max, dy)
        return cls(data=data, **kwargs)


class ZAxis(CoordinateAxis):
    data: Data[Z, float]

    @classmethod
    def init_from_limits(cls, z_min: float, z_max: float, dz: float, **kwargs):
        data = np.arange(z_min, z_max, dz)
        return cls(data=data, **kwargs)


@dataclass
class LongitudeAxis(CoordinateAxis):
    data: Data[LON, np.float32]
    name: Name[str] = "lon"
    standard_name: Attr[str] = "longitude"
    long_name: Attr[str] = "Longitude"
    units: Attr[str] = "degrees_east"

    @classmethod
    def init_from_limits(cls, lon_min: float, lon_max: float, dlon: float, **kwargs):
        data = np.arange(lon_min, lon_max, dlon)
        return cls(data=data, **kwargs)


@dataclass
class LatitudeAxis(CoordinateAxis):
    data: Data[LAT, np.float32]
    name: Name[str] = "lat"
    standard_name: Attr[str] = "latitude"
    long_name: Attr[str] = "Latitude"
    units: Attr[str] = "degrees_west"

    @classmethod
    def init_from_limits(cls, lat_min: float, lat_max: float, dlat: float, **kwargs):
        data = np.arange(lat_min, lat_max, dlat)
        return cls(data=data, **kwargs)


@dataclass
class TimeAxis:
    data: Data[TIME, Literal["datetime64[ns]"]]
    name: Name[str] = "time"
    long_name: Attr[str] = "Date"

    @classmethod
    def init_from_limits(cls, t_min: str, t_max: str, dt: str, **kwargs):
        t_min = pd.to_datetime(t_min)
        t_max = pd.to_datetime(t_max)
        dt = pd.to_timedelta(dt)
        data = np.arange(t_min, t_max, dt)
        return cls(data=data, **kwargs)

    @property
    def ndim(self):
        return len(self.data)


@dataclass
class Grid2D(AsDataArray):
    lon: Coordof[LongitudeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0

    @property
    def ndim(self):
        return (self.lat.ndim, self.lon.ndim)
    
    @property
    def spatial_grid(self):
        grid = np.meshgrid(self.lat.data, self.lon.data, indexing="ij")
        return np.stack(grid, axis=-1)
    

@dataclass
class Grid2DT(AsDataArray):
    data: Data[tuple[TIME, LAT, LON], np.float32]
    time: Coordof[TimeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "var"

    @property
    def ndim(self):
        return (self.time.ndim, self.lat.ndim, self.lon.ndim)
        


@dataclass
class SSH2D:
    data: Data[tuple[LAT, LON], np.float32]
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "ssh"
    units: Attr[str] = "m"
    standard_name: Attr[str] = "sea_surface_height"
    long_name: Attr[str] = "Sea Surface Height"

    @property
    def ndim(self):
        return (self.lat.ndim, self.lon.ndim)

    @classmethod
    def init_from_axis(cls, lon: LongitudeAxis, lat: LatitudeAxis):
        data_init = np.ones((lat.ndim, lon.ndim))
        return cls(data=data_init, lon=lon, lat=lat)
    
    @classmethod
    def init_from_grid(cls, grid: Grid2D):
        return cls(data=np.ones(grid.ndim), lon=grid.lon, lat=grid.lat)


@dataclass
class SSH2DT:
    data: Data[tuple[TIME, LAT, LON], np.float32]
    time: Coordof[TimeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "ssh"
    units: Attr[str] = "m"
    standard_name: Attr[str] = "sea_surface_height"
    long_name: Attr[str] = "Sea Surface Height"

    @property
    def ndim(self):
        return (self.time.ndim, self.lat.ndim, self.lon.ndim)

    @classmethod
    def init_from_axis(cls, lon: LongitudeAxis, lat: LatitudeAxis, time: TimeAxis):
        return cls(
            data=np.ones((time.ndim, lat.ndim, lon.ndim)),
            time=time, lon=lon, lat=lat)
    
    @classmethod
    def init_from_grid(cls, grid: Grid2DT):
        return cls(
            data=np.ones(grid.ndim),
            lon=grid.lon, lat=grid.lat, time=grid.time)
