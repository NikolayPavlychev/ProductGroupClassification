

print('Import libs...')

import time
time_start = time.time()

import os
import sys
import random
import re
import joblib
import json

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(42)
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score,recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import pymorphy2
# from natasha import (
#     Segmenter,
#     MorphVocab,
    
#     NewsEmbedding,
#     NewsMorphTagger,
#     NewsSyntaxParser,
#     NewsNERTagger,
    
#     PER,
#     NamesExtractor,

#     Doc
# )


import parapply
import importlib
importlib.reload(parapply)
from parapply import parapply

from tqdm import tqdm
tqdm.pandas()

import itertools
from itertools import product

import seaborn as sns



# import functions
# import importlib
# importlib.reload(functions)
# from functions import *





dataset_sub = pd.read_csv(ROOT_DIR + '/dataset_groups_preprocessing_100000.csv',sep=';',dtype=object)

dataset_sub = dataset_sub.dropna()
print(dataset_sub.shape)
dataset_sub = dataset_sub.reset_index(drop=True)

dataset_sub.head(10)

# dataset_sub_features.to_csv(ROOT_DIR + '/dataset_sub_features.csv',sep=';',index=False)
dataset_sub_features = pd.read_csv(ROOT_DIR + '/dataset_sub_features.csv',sep=';')



enc = LabelEncoder()
dataset_sub['target'] = enc.fit_transform(dataset_sub['group_code'])
dataset_sub_stats = pd.DataFrame(dataset_sub['target'].value_counts()).reset_index()
dataset_sub_stats.head(10)

dataset_sub[dataset_sub['target']==166]

dataset_sub_stats = dataset_sub_stats.rename(columns={'index':'target','target':'count'})
dataset_sub_stats.columns

dataset_sub['target'].value_counts()

dataset_sub['brend_code'].unique().shape

import pandas as pd

def OhePreprocessing(dataset, target=True, train_bool=True, cat_dummies = None, train_cols_order = None):
    cols=list(dataset.columns)

    if target:
        dataset_ohe_form = dataset[['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'DEBT']]
    else:
        dataset_ohe_form = dataset[['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID']]

    cols.remove('ISU')
    cols.remove('ST_YEAR')
    cols.remove('SEMESTER')
    cols.remove('DISC_ID')
    # cols.remove('TYPE_NAME')
    if 'DEBT' in cols:
        cols.remove('DEBT')

    for col in cols:
        print(col)
        if pd.api.types.is_object_dtype(dataset[col]):
            df = pd.get_dummies(dataset[col], prefix=str(col), prefix_sep="__",
                              columns=dataset[col])
        else:
            df = pd.DataFrame(dataset[col])
        dataset_ohe_form = pd.concat((dataset_ohe_form,df),axis=1)


    if train_bool:
        cat_dummies = [col for col in dataset_ohe_form
                   if "__" in col
                   and col.split("__")[0] in cols]
        train_cols_order = list(dataset_ohe_form.columns)

    else:
        for col in dataset_ohe_form.columns:
            if ("__" in col) and (col.split("__")[0] in cols) and col not in cat_dummies:
                print("Removing additional feature {}".format(col))
                dataset_ohe_form.drop(col, axis=1, inplace=True)

        for col in cat_dummies:
            if col not in dataset_ohe_form.columns:
                print("Adding missing feature {}".format(col))
                dataset_ohe_form[col] = 0

        if target:
            dataset_ohe_form = dataset_ohe_form[train_cols_order]
        else:
            train_cols_order.remove('DEBT')


    if train_bool:
        return dataset_ohe_form, cat_dummies, train_cols_order
    else:
        return dataset_ohe_form[train_cols_order]

dataset_sub.dtypes

dataset_sub_brand_ohe = pd.get_dummies(dataset_sub['brend_code'],prefix='brand')
dataset_sub_brand_ohe.head(5)

print(dataset_sub.shape,dataset_sub_brand_ohe.shape,dataset_sub_features.shape)

print(dataset_sub_brand_ohe.dtypes, dataset_sub_features.iloc[:,4:].dtypes)

dataset_train_test = pd.concat((pd.DataFrame(dataset_sub['target']), dataset_sub_brand_ohe, dataset_sub_features.iloc[:,4:]),axis=1)
dataset_train_test['target'] = dataset_train_test['target'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(dataset_train_test.drop(['target'],axis=1),dataset_train_test['target'],test_size=0.3,random_state=42)

X_train.head(5)

y_train.dtypes

y_train.shape

lr = LogisticRegression(C=1,random_state=42,n_jobs=-1,class_weight='balanced')
lr.fit(X_train, y_train)

y_test_val = pd.DataFrame(y_test,columns=['target'])
# y_test_val = y_test_val.drop(['predict'],axis=0)
y_test_val['predict'] = lr.predict(X_test)

y_train_val = pd.DataFrame(y_train,columns=['target'])
# y_train_val = y_train_val.drop(['predict'],axis=0)
y_train_val['predict'] = lr.predict(X_train)

report = classification_report(y_test_val['target'], y_test_val['predict'], output_dict=True)
test_report = pd.DataFrame(report).transpose()

report = classification_report(y_train_val['target'], y_train_val['predict'], output_dict=True)
train_report = pd.DataFrame(report).transpose()

test_report.sort_values(by='f1-score',ascending=False)

train_report.sort_values(by='f1-score',ascending=False)

y_test_val.dtypes

y_test_val['predict'].value_counts()

y_test_val['target'].value_counts()

test_report.to_excel(ROOT_DIR+ '/test_report.xlsx',index=False)
train_report.to_excel(ROOT_DIR+ '/train_report.xlsx',index=False)

accuracy_test = accuracy_score(y_test_val['target'], y_test_val['predict'])
accuracy_train = accuracy_score(y_train_val['target'], y_train_val['predict'])
print('accuracy_test=',accuracy_test,'accuracy_train=',accuracy_train)

y_test_val['target'].value_counts()

