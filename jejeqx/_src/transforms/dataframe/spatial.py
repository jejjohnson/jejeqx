from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from jejeqx._src.transforms.spatial import (
    spherical_to_cartesian_3d,
    cartesian_to_spherical_3d,
)
from metpy.calc import lat_lon_grid_deltas


class LatLonDeg2Meters(BaseEstimator, TransformerMixin):
    def __init__(self, units: str = "degrees"):
        self.units = units

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray, y=None):
        pass

    def inverse_transform(self, X: np.ndarray, y=None):
        msg = "This method is not invertible..."
        raise NotImplementedError(msg)


class Spherical2Cartesian(BaseEstimator, TransformerMixin):
    def __init__(self, radius: float = 6371.010, units: str = "degrees"):
        self.radius = radius
        self.units = units

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        lon, lat = X["lon"].values, X["lat"].values

        if self.units == "degrees":
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)

        x, y, z = spherical_to_cartesian_3d(lon=lon, lat=lat, radius=self.radius)

        X = np.stack([x, y, z], axis=-1)
        X = pd.DataFrame(X, columns=["x", "y", "z"])

        return X

    def inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:

        lon, lat, _ = cartesian_to_spherical_3d(
            x=X["x"],
            y=X["y"],
            z=X["z"],
        )

        if self.units == "degrees":
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)

        X = np.stack([lat, lon], axis=-1)

        X = pd.DataFrame([lat, lon], columns=["lat", "lon"])

        return X


class Cartesian2Spherical(Spherical2Cartesian):
    def __init__(self, radius: float = 6371.010):
        super().__init__(radius=radius)

    def transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:

        X = super().inverse_transform(X=X, y=y)

        return X

    def inverse_transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:

        X = super().transform(X=X, y=y)

        return X
