from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from jejeqx._src.transforms.spatial import spherical_to_cartesian_3d, cartesian_to_spherical_3d


class Spherical2Cartesian(BaseEstimator, TransformerMixin):
    def __init__(self, radius: float=6371.010, units: str="degrees"):
        self.radius = radius
        self.units = units
        
    def fit(self, X: np.ndarray, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
                
        lon, lat = X["lon"], X["lat"]
        
        if self.units == "degrees":
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)
            
        X["x"], X["y"], X["z"] = spherical_to_cartesian_3d(
            lon=lon, lat=lat, radius=self.radius
        )
        
        return X
    
    def inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        
        
        lon, lat, _ = cartesian_to_spherical_3d(
            x=X["x"], y=X["y"], z=X["z"],
        )
        
        if self.units == "degrees":
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)
        X["lon"], X["lat"] = lon, lat
        return X
    
    
class Cartesian2Spherical(Spherical2Cartesian):
    def __init__(self, radius: float=6371.010):
        super().__init__(radius=radius)
    

    
    def transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:
        
        X = super().inverse_transform(X=X, y=y)
        
        return X
    
    def inverse_transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:
        
        X = super().transform(X=X, y=y)
        
        return X
            
        