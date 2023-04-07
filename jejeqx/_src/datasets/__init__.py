import torch.utils.data as data


class RegressionDataset(data.Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class SpatioTempDataset(data.Dataset):
    def __init__(self, space_coords, time_coords, data=None):
        super().__init__()
        self.space_coords = space_coords
        self.time_coords = time_coords
        self.data = data
        
    def __len__(self):
        return self.space_coords.shape[0]
    
    def __getitem__(self, idx):
        outputs = dict()
        
        outputs["spatial"] = self.space_coords[idx]
        outputs["temporal"] = self.time_coords[idx]
        if self.data is not None:
            outputs["data"] = self.data[idx]
        
        return outputs
    

class SpatioTempParamDataset(data.Dataset):
    def __init__(self, x_space, x_time, x_params, y):
        super().__init__()
        self.x_space = x_space
        self.x_time = x_time
        self.x_params = x_params
        self.y = y
        
    def __len__(self):
        return self.x_space.shape[0]
    
    def __getitem__(self, idx):
        outputs = dict()
        
        outputs["spatial"] = self.x_space[idx]
        outputs["temporal"] = self.x_time[idx]
        outputs["params"] = self.x_params[idx]
        outputs["y"] = self.y[idx]
        
        return outputs
