

import numpy as np
import pandas as pd



class OHEncoder(object):
    
    def __init__(self, cat_cols=None):
        
        self.cat_cols = cat_cols
        
        
        
    def transform(self, data):
        
        data_out = data.copy()
        
        if self.cat_cols == None:
            self.cat_cols = data.select_dtypes('object').columns.tolist()
        
        data_out = pd.get_dummies(data_out, prefix=self.cat_cols, columns=self.cat_cols)
        
        uint_cols = data_out.select_dtypes('uint8').columns.tolist()
        data_out[uint_cols] = data_out[uint_cols].astype(int)
        
        return data_out
    
    
    
    def inverse_transform(self, data):
        
        data_out = data.copy()
        
        drop_cols = []
        for ii in self.cat_cols:
            
            oh_cols = [x for x in data_out.columns if ii + '_' in x]
            class_names = [x.replace(ii + '_', '') for x in oh_cols]
            
            col_mapper = dict(zip(oh_cols, class_names))
            
            data_out[ii] = data_out[oh_cols].rename(mapper=col_mapper, axis=1).idxmax(axis=1)
            drop_cols += oh_cols
        
        data_out.drop(drop_cols, axis=1, inplace=True)
        
        return data_out