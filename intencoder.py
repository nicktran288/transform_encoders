

# Author: Wah (Nick) Tran
# Status: Production


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder



class IntegerEncoder(object):
    '''
    Transformer to encode multiple categorical features as integers.

    Applies sklearn's LabelEncoder to dtype:object columns to produce sequential 
    integers starting at 0 to represent unique classes. Encodes null values as -999.
    '''    

    def __init__(self, cat_cols=None):
        '''
        Parameters
        ----------

        cat_cols : list or None (default=None)
            Column names to be encoded.
            If None, will default to all columns identified as dtype:object.

        Attributes
        ----------

        cat_cols : list
            Column names to be encoded.

        n_cat_cols: int
            Number of categorical columns to encode.

        encoders: dict
            Container for LabelEncoders fit to each column in cat_cols.
        '''

        self.cat_cols = cat_cols
    
    
    
    def fit(self, data):
        '''
        Fit encoder to dataset.

        Parameters
        ----------

        data: DataFrame, shape (n_samples, n_features)
            Full dataset for which categorical features are to be identified.

        Returns
        -------

        self: returns an instance of self.

        '''

        # If no columns are specified, all dtype:object columns will be encoded
        if self.cat_cols==None:
            self.cat_cols = data.select_dtypes('object').columns.tolist()
            
        self.n_cat_cols = len(self.cat_cols)
        print('\n{} categorial features detected\n'.format(self.n_cat_cols))
        

        # Fit LabelEncoder to each column in cat_cols and store encoder
        self.encoders = {}
        for i, ii in enumerate(self.cat_cols):
            cat_data = data[ii].astype(str).str.lower().replace('nan', np.nan)
            non_na_ind = cat_data[~(cat_data.isna())].index
            
            le = LabelEncoder()
            le.fit(cat_data.loc[non_na_ind])
            
            self.encoders[ii] = le
            
            print('{:<8} scanned: "{}"'.format('({}/{})'.format(i+1, self.n_cat_cols), ii))
            
        return self
          
        
       
    def transform(self, data):
        '''
        Transform class labels to integer labels

        Parameters
        ----------

        data: DataFrame, shape (n_samples, n_features)
            Full dataset for which categorical features are to be encoded.

        Returns
        -------

        data_out: DataFrame, shape (n_samples, n_features)
            Full dataset with categorical features encoded as integers and
            null values encoded as -999.
        '''

        data_out = data.copy()
        
        for i, ii in enumerate(self.cat_cols):
            cat_data = data_out[ii].astype(str).str.lower().replace('nan', np.nan)
            
            non_na_ind = cat_data[~(cat_data.isna())].index
            cat_data.loc[non_na_ind] = self.encoders[ii].transform(cat_data.loc[non_na_ind])
            
            data_out[ii] = cat_data
            
            print('{:<8} transformed: "{}"'.format('({}/{})'.format(i+1, self.n_cat_cols), ii))
        

        print('\nEncoding null values...')
        data_out[self.cat_cols] = data_out[self.cat_cols].fillna(-999).astype(int)
        
        return data_out

       
        
    def fit_transform(self, data):
        '''
        Fit encoder and return encoded data.

        Parameters
        ----------

        data: DataFrame, shape (n_samples, n_features)
            Full dataset for which categorical features are to be identified and encoded.

        Returns
        -------

        data_out: DataFrame, shape (n_samples, n_features)
            Full dataset with categorical features encoded as integers and
            null values encoded as -999.
        '''
        
        return self.fit(data).transform(data)
    
    
    
    def inverse_transform(self, data):
        '''
        Returns integer values to original classes and -999 values to nulls.

        Parameters
        ----------

        data: DataFrame, shape (n_samples, n_features)
            Full dataset for which categorical features are to be returned to 
            their original calues.
            
        Returns
        -------

        data_out: DataFrame, shape (n_samples, n_features)
            Full dataset with categorical features  returned to orginal values.
        '''
        
        data_out = data.copy()
        
        print('\nRecoding null values...')
        data_out[self.cat_cols] = data_out[self.cat_cols].replace(-999, np.nan)

        
        for i, ii in enumerate(self.cat_cols):
            cat_data = data_out[ii]
            
            non_na_ind = cat_data[~(cat_data.isna())].index
            cat_data.loc[non_na_ind] = self.encoders[ii].inverse_transform(cat_data.loc[non_na_ind].astype(int))
            
            data_out[ii] = cat_data
            
            print('({}/{}) transformed: "{}"'.format(i+1, self.n_cat_cols, ii))
        
        return data_out
        
        
        