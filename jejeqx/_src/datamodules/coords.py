from typing import List, Dict, Optional, Callable
import warnings
import xarray as xr
import pandas as pd
import pytorch_lightning as pl
from jejeqx._src.datasets import SpatioTempDataset
from jejeqx._src.dataloaders import NumpyLoader
from jejeqx._src.transforms.spatial import validate_lon, validate_lat
from jejeqx._src.transforms.temporal import decode_cf_time, validate_time
from sklearn.model_selection import train_test_split
from dask.array.core import PerformanceWarning


class AlongTrackDM(pl.LightningDataModule):
    def __init__(self,
                 paths: List[str],
                 spatial_coords: List[str]=["lat", "lon"],
                 temporal_coords: List[str]=["time"],
                 variables: List[str]=["ssh"],
                 batch_size: int=128,
                 select: Dict=None,
                 iselect: Dict=None,
                 coarsen: Dict=None,
                 resample: Dict=None,
                 spatial_transform: Callable=None,
                 temporal_transform: Callable=None,
                 variable_transform: Callable=None,
                 shuffle: bool=True,
                 split_seed: int=123,
                 train_size: float=0.8,
                 subset_size: Optional[int]=None,
                 subset_seed: int=42,
                 time_units: str='seconds since 2012-10-01',
                ):
        super().__init__()

        self.paths = paths
        self.spatial_coords = spatial_coords
        self.temporal_coords = temporal_coords
        self.variables = variables
        self.batch_size = batch_size
        self.select = select
        self.iselect = iselect
        self.coarsen = coarsen
        self.resample = resample
        self.split_seed = split_seed
        self.train_size = train_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.variable_transform = variable_transform
        self.shuffle = shuffle
        self.time_units = time_units
        
        
        
    def load_xrds(self, paths=None, **kwargs):
        
        if paths is None:
            paths = self.paths
        
        def preprocess(ds):
            ds = validate_time(ds)
            ds = validate_lon(ds)
            ds = validate_lat(ds)
            ds = ds.sortby("time")
            
            if self.select is not None:
                ds = ds.sel(**self.select)
            if self.iselect is not None:
                ds = ds.isel(**self.iselect)
            if self.coarsen is not None:
                ds = ds.coarsen(dim=self.coarsen, boundary="trim").mean()
            if self.resample is not None:
                try:
                    ds = ds.resample(time="1D").mean()
                except IndexError:
                    pass
                
            ds = decode_cf_time(ds, units=self.time_units)
                
            return ds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            # Note: there is an annoying performance memory due to the chunking

            ds = xr.open_mfdataset(
                paths=paths, preprocess=preprocess, 
                combine="nested",
                concat_dim="time",
                **kwargs)
            
            ds = ds.sortby("time")

            return ds.compute()

    def preprocess(self):

        ds = self.load_xrds(paths=self.paths)
        
        # convert xarray to daraframe
        ds = ds.to_dataframe()
        
        ds = ds.dropna()
        
        # extract coordinates (for later)
        self.coord_index = ds.index
        
        # remove the indexing to get single columns
        ds = ds.reset_index()

        column_names = ds.columns.values

        msg = f"No requested spatial coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.spatial_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.spatial_coords).intersection(column_names)) == len(self.spatial_coords), msg

        msg = f"No requested temporal coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.temporal_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.temporal_coords).intersection(column_names)) == len(self.temporal_coords), msg

        msg = f"No requested variables found in dataset:"
        msg += f"\nVariables: {self.variables}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.variables).intersection(column_names)) == len(self.variables), msg


        x = ds[self.spatial_coords]
        t = ds[self.temporal_coords]
        y = ds[self.variables]
        
        # do specific spatial-temporal-variable transformations
        if self.spatial_transform is not None:
            x = self.spatial_transform.fit_transform(x)
        if self.temporal_transform is not None:
            t = self.temporal_transform.fit_transform(t)
        if self.variable_transform is not None:
            y = self.variable_transform.fit_transform(y)
            
        
        # extract the values
        x, t, y = x.values, t.values, y.values

        self.spatial_dims = x.shape[-1]
        self.temporal_dims = t.shape[-1]
        self.variable_dims = y.shape[-1]
        
        return x, t, y
    
    def setup(self, stage=None):
        
        x, t, y = self.preprocess()
        
        self.ds_test = SpatioTempDataset(x, t, y)
        
        if self.subset_size is not None:
            x, t, y = self.subset(x, t, y)
        
        # train/validation/test split
        xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = self.split(x, t, y)
        
        # create spatial-temporal datasets
        self.ds_train = SpatioTempDataset(xtrain, ttrain, ytrain)
        self.ds_valid = SpatioTempDataset(xvalid, tvalid, yvalid)
        

    def subset(self, x, t, y):
        
        x, _, t, _, y, _ = train_test_split(
            x, t, y, 
            train_size=self.subset_size, 
            random_state=self.subset_seed, 
            shuffle=True
        )
        
        return x, t, y
        
    def split(self, x, t, y):
        
        xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = train_test_split(
            x, t, y, 
            train_size=self.train_size, 
            random_state=self.split_seed, 
            shuffle=True
        )
        return xtrain, xvalid, ttrain, tvalid, ytrain, yvalid

    def data_to_df(self, x):
        return pd.DataFrame(x, index=self.coord_index, columns=self.variables)
    
    def train_dataloader(self):
        return NumpyLoader(self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return NumpyLoader(self.ds_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)
        
        
class EvalCoordDM(AlongTrackDM):
    
    

    
    def setup(self, stage=None):
        
        x, t, y = self.preprocess()

        self.ds_test = SpatioTempDataset(x, t, y)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()
    
    def test_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)
    
    
class EvalGridDM(pl.LightningDataModule):
    def __init__(self,
                 spatial_coords: List[str]=["lat", "lon"],
                 temporal_coords: List[str]=["time"],
                 variables: List[str]=["ssh"],
                 batch_size: int=128,
                 select: Dict=None,
                 iselect: Dict=None,
                 coarsen: Dict=None,
                 resample: Dict=None,
                 spatial_transform: Callable=None,
                 temporal_transform: Callable=None,
                 variable_transform: Callable=None,
                 shuffle: bool=True,
                 split_seed: int=123,
                 train_size: float=0.8,
                 subset_size: Optional[int]=None,
                 subset_seed: int=42,
                 time_units: str='seconds since 2012-10-01',
                ):
        super().__init__()

        self.paths = paths
        self.spatial_coords = spatial_coords
        self.temporal_coords = temporal_coords
        self.variables = variables
        self.batch_size = batch_size
        self.select = select
        self.iselect = iselect
        self.coarsen = coarsen
        self.resample = resample
        self.split_seed = split_seed
        self.train_size = train_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.variable_transform = variable_transform
        self.shuffle = shuffle
        self.time_units = time_units
        
        
    def preprocess(self):
        
         x, y, z = np.meshgrid(
            ds.coords["time"].data,
            ds.coords["latitude"].data,
            ds.coords["longitude"].data,
        )

        ds = self.load_xrds(paths=self.paths)
        
        # convert xarray to daraframe
        ds = ds.to_dataframe()
        
        ds = ds.dropna()
        
        # extract coordinates (for later)
        self.coord_index = ds.index
        
        # remove the indexing to get single columns
        ds = ds.reset_index()

        column_names = ds.columns.values

        msg = f"No requested spatial coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.spatial_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.spatial_coords).intersection(column_names)) == len(self.spatial_coords), msg

        msg = f"No requested temporal coordinates found in dataset:"
        msg += f"\nTemporal Coords: {self.temporal_coords}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.temporal_coords).intersection(column_names)) == len(self.temporal_coords), msg

        msg = f"No requested variables found in dataset:"
        msg += f"\nVariables: {self.variables}"
        msg += f"\nColumns: {column_names}"
        assert len(set(self.variables).intersection(column_names)) == len(self.variables), msg

        x = ds[self.spatial_coords]
        t = ds[self.temporal_coords]
        y = ds[self.variables]
        
        # do specific spatial-temporal-variable transformations
        if self.spatial_transform is not None:
            x = self.spatial_transform.fit_transform(x)
        if self.temporal_transform is not None:
            t = self.temporal_transform.fit_transform(t)
        if self.variable_transform is not None:
            y = self.variable_transform.fit_transform(y)
            
        
        # extract the values
        x, t, y = x.values, t.values, y.values

        self.spatial_dims = x.shape[-1]
        self.temporal_dims = t.shape[-1]
        self.variable_dims = y.shape[-1]
        
        return x, t, y
    
    def setup(self, stage=None):
        
                
       
        
        x, t, y = self.preprocess()

        self.ds_test = SpatioTempDataset(x, t, y)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()
    
    def test_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return NumpyLoader(self.ds_test, batch_size=self.batch_size)
        
    