#Function OnePreprocessing() convert dataset, contain categorical features and transform them into OHE form.
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

def OhePreprocessing(dataset,engine,prefix_name): #cols_order = None):
   
    ROOT_DIR = os.path.abspath(os.curdir)

    if engine =='train':

        enc = OneHotEncoder(sparse=True)
        enc.fit(dataset['brend_code'].to_numpy().reshape(-1, 1))
        joblib.dump(enc, ROOT_DIR+ '/'+prefix_name+'_ohe_encoder.pickle')

        cat_dummies = enc.transform(dataset['brend_code'].to_numpy().reshape(-1, 1))
        cols = []
        for col in enc.categories_[0]:
            col = 'brend_code__'+col
            cols.append(col)

    if engine =='inference':  
        enc = joblib.load(ROOT_DIR+ '/'+prefix_name+'_ohe_encoder.pickle')

        cat_dummies = enc.transform(dataset['brend_code'].to_numpy().reshape(-1, 1))
        cols = []
        for col in enc.categories_[0]:
            col = 'brend_code__'+col
            cols.append(col)

    if engine =='inference_production':  
        enc = joblib.load(ROOT_DIR+ '/ohe_encoder.pickle')
        cat_dummies = enc.transform(dataset.to_numpy().reshape(-1, 1))
        cols = []
        for col in enc.categories_[0]:
            col = 'brend_code__'+col
            cols.append(col)
    return cat_dummies, cols