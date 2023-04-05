from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MinMaxDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]=None, min_val: float=-1, max_val: float=1):
        self.columns = columns
        self.transformer = MinMaxScaler((min_val, max_val))
        
    def fit(self, X: pd.DataFrame, y=None):

        if self.columns is None:
            self.columns = X.columns.data
        
        X_var = X[self.columns].values
        
        self.transformer.fit(X_var)
        
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        
        X_var = X[self.columns].values
        
        X_var = self.transformer.transform(X_var)
        
        X[self.columns] = X_var
        
        return X
    
    def inverse_transform(self, X: pd.DataFrame, y=None):
        
        X_var = X[self.columns].values
        
        X_var = self.transformer.inverse_transform(X_var)
        
        X[self.columns] = X_var
        
        return X
        
        