from typing import Dict, List, Callable
import pytorch_lightning as pl
from pathlib import Path
from jejeqx._src.utils.io import runcmd
import pandas as pd
from jejeqx._src.dataloaders import NumpyLoader
import xarray as xr
import numpy as np
from jejeqx._src.datasets import RegressionDataset, SpatioTempDataset
from jejeqx._src.dataloaders import NumpyLoader
from sklearn.model_selection import train_test_split


FILE_GULFSTREAM_SSH_NATL60 = "https://s3.us-east-1.wasabisys.com/melody/osse_data/ref/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc"
FILE_GULFSTREAM_SSH_SWOT = "https://s3.us-east-1.wasabisys.com/melody/osse_data/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc"
FILE_GULFSTREAM_SSH_DUACS = "https://s3.us-east-1.wasabisys.com/melody/osse_data/oi/ssh_NATL60_swot_4nadir.nc"
FILE_GULFSTREAM_SST_NATL60 = "https://s3.us-east-1.wasabisys.com/melody/osse_data/ref/NATL60-CJM165_GULFSTREAM_sst_y2013.1y.nc"
FILE_GULFSTREAM_SSS_NATL60 = "https://s3.us-east-1.wasabisys.com/melody/osse_data/ref/NATL60-CJM165_GULFSTREAM_sss_y2013.1y.nc"
FILE_GULFSTREAM_SSH_NADIR = "https://s3.us-east-1.wasabisys.com/melody/osse_data/data/gridded_data_swot_wocorr/dataset_nadir_0d.nc"
FILE_GULFSTREAM_SSH_OI_NADIR = "https://s3.us-east-1.wasabisys.com/melody/osse_data/oi/ssh_NATL60_4nadir.nc"
FILE_GULFSTREAM_SSH_SWOT = "https://s3.us-east-1.wasabisys.com/melody/osse_data/data/gridded_data_swot_wocorr/dataset_swot.nc"
FILE_GULFSTREAM_SSH_OI_SWOT = "https://s3.us-east-1.wasabisys.com/melody/osse_data/oi/ssh_NATL60_swot.nc"


class XRDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        directory: str="./",
        batch_size: int=32, 
        shuffle: bool=False, 
        split_method: str="even",
        download: bool=True,
        select: Dict=None,
        iselect: Dict=None,
        coarsen: Dict=None,
        resample: Dict=None,
        coords: List=["lat", "lon"],
        variables: List=["ssh"],
        train_size: float=0.80,
        random_state: float=123,
        transforms: Callable=None,
    ):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.split_method = split_method
        self.shuffle = shuffle
        self.download = download
        self.select = select
        self.iselect = iselect
        self.coarsen = coarsen
        self.resample = resample
        self.coords = coords
        self.variables = variables
        self.transforms = transforms
        self.train_size = train_size
        self.random_state = random_state
        
    def load_xrdata(self):
        if isinstance(self.file_name, str):
            path = Path(self.directory).joinpath(self.file_name)
        
            if path.is_file():
                return self._load_data(path)
            else:
                raise ValueError(f"file doesnt exist: {path}")
        elif isinstance(self.file_name, list):
            
            datasets = list()
            for ifile in self.file_name:
                path = Path(self.directory).joinpath(ifile)
                
                
                if path.is_file():
                    ids = self._load_data(path)
                    print(ids)
                    datasets.append(ids)
                else:
                    raise ValueError(f"file doesnt exist: {path}")
            
            # dims = list(datasets[0].coords.keys())
            datasets = xr.merge(datasets, join="inner")
            
            datasets = datasets.transpose("time", "lat", "lon")
            print(datasets)
            
            return datasets
                
#         elif not path.is_file() and self.download:
#             cmd = f"wget -nc "
#             cmd += f"--directory-prefix={self.directory} "
#             cmd += f"{FILE_NATL60_GULFSTREAM}"
            
#             runcmd(cmd, verbose=True)
            
#             return self._load_data(path)
        
        else:
            raise ValueError(f"Unrecognized type...")
            
    
    def _load_data(self, path, decode: bool=False):
        
        
        def preprocess(ds):
            
            if self.select is not None:
                ds = ds.sel(**self.select)
            if self.iselect is not None:
                ds = ds.isel(**self.iselect)
            if self.coarsen is not None:
                ds = ds.coarsen(dim=self.coarsen, boundary="trim").mean()
            if self.resample is not None:
                ds = ds.resample(time=self.resample).mean()
            return ds
                
        

        ds = xr.open_dataset(
                path, decode_times=False
            ).assign_coords(time=lambda ds: pd.to_datetime(ds.time))
        
        if self.select is not None:
            ds = ds.sel(**self.select)
        if self.iselect is not None:
            ds = ds.isel(**self.iselect)
        if self.coarsen is not None:
            ds = ds.coarsen(dim=self.coarsen, boundary="trim").mean()
            
        return ds.compute()
        
    def setup(self, stage=None):
        # load
        
        ds = self.load_xrdata()
        
        # convert xarray to daraframe
        ds = ds.to_dataframe()
        
        ds = ds.dropna()
        
        self.coord_index = ds.index
        
        ds = ds.reset_index()
        
        
        
        
        if self.transforms is not None:
            ds = self.transforms.fit_transform(ds)
        
        x = ds[self.coords].values
        y = ds[self.variables].values
        xtrain, xvalid, ytrain, yvalid = self.split(x, y)
        
        self.ds_train = RegressionDataset(xtrain, ytrain)

        self.ds_valid = RegressionDataset(xvalid, yvalid)

        self.ds_test = RegressionDataset(x, y)

        return self
    
        
    def split(self, x, y):
        
        if self.split_method == "even":
            xtrain, ytrain = x[::2], y[::2]
            xvalid, yvalid = x[1::2], y[1::2]
        elif self.split_method == "random":
            xtrain, xvalid, ytrain, yvalid = train_test_split(
                x, y, 
                train_size=self.train_size, 
                random_state=self.random_state, 
                shuffle=True
            )
        else:
            raise ValueError(f"Unrecognized split method")
            
        return xtrain, xvalid, ytrain, yvalid
    
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
    
    

    
class XRSTDataModule(XRDataModule):
    
    def __init__(
        self, 
        directory: str="./",
        batch_size: int=32, 
        shuffle: bool=False, 
        split_method: str="even",
        download: bool=True,
        select: Dict=None,
        iselect: Dict=None,
        coarsen: Dict=None,
        spatial_coords: List=["lat", "lon"],
        temporal_coords: List=["time"],
        variables: List=["ssh"],
        train_size: float=0.80,
        random_state: float=123,
        transforms: Callable=None,
    ):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.split_method = split_method
        self.shuffle = shuffle
        self.download = download
        self.select = select
        self.iselect = iselect
        self.coarsen = coarsen
        self.spatial_coords = spatial_coords
        self.temporal_coords = temporal_coords
        self.variables = variables
        self.transforms = transforms
        self.train_size = train_size
        self.random_state = random_state
        
            
    def setup(self, stage=None):
        # load
        
        ds = self.load_xrdata()        

            
        # convert xarray to daraframe
        ds = ds.to_dataframe()
        
        ds = ds.dropna()
        
        self.coord_index = ds.index
        
        ds = ds.reset_index()
        
        if self.transforms is not None:
            ds = self.transforms.fit_transform(ds)
        
        x = ds[self.spatial_coords].values
        t = ds[self.temporal_coords].values
        y = ds[self.variables].values
        
        xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = self.split(x, t, y)
        
        self.ds_train = SpatioTempDataset(xtrain, ttrain, ytrain)

        self.ds_valid = SpatioTempDataset(xvalid, tvalid, yvalid)

        self.ds_test = SpatioTempDataset(x, t, y)


        return self
    
        
    def split(self, x, t, y):
        
        if self.split_method == "even":
            xtrain, ttrain, ytrain = x[::2], t[::2], y[::2]
            xvalid, tvalid, yvalid = x[1::2], t[1::2], y[1::2]
        elif self.split_method == "random":
            xtrain, xvalid, ttrain, tvalid, ytrain, yvalid = train_test_split(
                x, t, y, 
                train_size=self.train_size, 
                random_state=self.random_state, 
                shuffle=True
            )
        else:
            raise ValueError(f"Unrecognized split method")
            
        return xtrain, xvalid, ttrain, tvalid, ytrain, yvalid
    
    
    
class SSHNATL60(XRDataModule):
    file_name = "NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc"
    
    
class SSHSTNATL60(XRSTDataModule):
    file_name = "NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc"
    
class SSHSSTSTNATL60(XRSTDataModule):
    file_name = [
        "NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc",
        "NATL60-CJM165_GULFSTREAM_sst_y2013.1y.nc",
    ]
    
class SSHSTSWOT(XRSTDataModule):
    file_name = "dataset_nadir_0d_swot.nc"
    
    def _load_data(self, path):
        
        ds = xr.open_dataset(path)
        if self.select is not None:
            ds = ds.sel(**self.select)
        if self.iselect is not None:
            ds = ds.isel(**self.iselect)
        if self.coarsen is not None:
            ds = ds.coarsen(dim=self.coarsen, boundary="trim").mean()
            
        return ds.compute()