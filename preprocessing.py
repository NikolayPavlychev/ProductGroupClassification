#/ProductGroupClassification/preprocessing_model.py created by: Nikolay Pavlychev nikolaypavlychev@yandex.ru
#-----------------------------------------------------------------------------------------------------------------------

#/home/pavlychev/anaconda3/bin/python /home/pavlychev/product_group_prediction/ProductGroupClassification/production/preprocessing.py dataset_groups_before_analyse.csv

print('Import libs')

import time
time_start = time.time()

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
import re
import joblib
import json
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import use
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import pymorphy2

import parapply
import importlib
importlib.reload(parapply)
from parapply import parapply

from tqdm import tqdm
tqdm.pandas()

import itertools
from itertools import product

import seaborn as sns

import functions
import importlib
importlib.reload(functions)
from functions import *

import OHE
importlib.reload(OHE)
from OHE import *


print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Import data')
time_start = time.time()

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

cols = ['artical', 'brend_code', 'desc', 'guid', 'group_code']
input_data_file = str(sys.argv[1])
print(input_data_file)
dataset = pd.read_csv(ROOT_DIR + '/' + input_data_file, dtype=object,sep=';', header=0,usecols=cols,nrows=1000)
prefix_name = input_data_file.split('.')[0]

print('dataset shape: ',dataset.shape)
print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Label encoding')
time_start = time.time()

if str(sys.argv[2]) =='train':

    enc = LabelEncoder()
    enc.fit(dataset['group_code'])
    joblib.dump(enc, ROOT_DIR+ '/'+prefix_name+'_label_encoder.pickle')
    dataset['target'] = enc.transform(dataset['group_code'])

    le_name_mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
    joblib.dump(le_name_mapping,ROOT_DIR+ '/'+prefix_name+'_target_group_code_mapping_full.pickle')

    brends_list = ['brend_code'+'__'+str(brend) for brend in list(dataset['brend_code'].unique())]

    with open(ROOT_DIR+ '/'+prefix_name+'_brends_list.txt', 'w') as f:
        f.write(' '.join(brends_list))
        f.close()


if str(sys.argv[2]) =='inference':

    enc = joblib.load(ROOT_DIR+ '/'+prefix_name+'_label_encoder.pickle')
    dataset['target'] = enc.transform(dataset['group_code'])

if str(sys.argv[2]) =='inference_production':

    enc = joblib.load(ROOT_DIR+ '/label_encoder.pickle')
    dataset['target'] = enc.transform(dataset['group_code'])


print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Filtering')
time_start = time.time()
dataset['desc'] = dataset['desc'].astype(str) 
dataset['desc'] = parapply(dataset['desc'], lambda x: voc_filter(x))

dataset.to_csv(ROOT_DIR + '/'+prefix_name+'_dataset_groups_preprocessing_filter.csv', sep=';', index=False)
print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------
print('Tokenization')
time_start = time.time()
dataset['desc_list'] = parapply(dataset['desc'], lambda x: x.split(' '))
print('process time: ',round(time.time()-time_start,2),' s')

dataset['desc_list_len'] = parapply(dataset['desc_list'], lambda x: len(x),n_jobs = 20,n_chunks=75)
dataset = dataset.sample(n=dataset.shape[0])
dataset = dataset.reset_index(drop=True)


rowcount  = 0
for row in open(ROOT_DIR +'/'+prefix_name+'_dataset_groups_preprocessing_filter.csv'):
  rowcount+= 1


# header = True

output_path = ROOT_DIR + '/'+prefix_name+'_dataset_train_test_preprocessing_lemma.csv'

for k, chunk in enumerate(pd.read_csv(ROOT_DIR +'/'+prefix_name+'_dataset_groups_preprocessing_filter.csv',sep=';',chunksize=rowcount//20)):
    print('\n')
    print('chunk ',k)
    print('Lemmatization')
    
    chunk['desc_list'] = parapply(chunk['desc'], lambda x: str(x).split(' '))
    chunk['desc_list_norm'] = parapply(chunk['desc_list'], lambda x: lemmatizer(x),n_jobs = 20,n_chunks=75)
    chunk['desc_clear_norm'] = parapply(chunk['desc_list_norm'], lambda x: ' '.join(x))

    chunk = chunk.dropna()

    print(chunk.shape[0]*(k+1))
 
    chunk.to_csv(output_path, sep='|',index=False, header=not os.path.exists(output_path), mode='a')
    # header = False
    if k==19:
        break

print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------
print('TF-IDF')
time_start = time.time()

dataset_train_test = pd.read_csv(ROOT_DIR + '/'+prefix_name+'_dataset_train_test_preprocessing_lemma.csv',sep='|',dtype=object)


if str(sys.argv[2]) =='train':

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vectorizer.fit(dataset_train_test['desc_clear_norm'])
    joblib.dump(ROOT_DIR+ '/'+prefix_name+'_tfidf_vectorizer.pickle')

    X_tfidf = tfidf_vectorizer.transform(dataset_train_test['desc_clear_norm'])
    X_tfidf_feature_filter_sum=X_tfidf.sum(axis=0)
    X_tfidf_feature_filter_sum=pd.DataFrame(np.transpose(np.array(X_tfidf_feature_filter_sum)),columns=['tf_idf'])
    X_tfidf=X_tfidf.tocsc(copy=False)

    X_tfidf_feature=pd.DataFrame(tfidf_vectorizer.get_feature_names())
    X_tfidf_feature=X_tfidf_feature.rename(columns={0:'vocabilary'})
    X_tfidf_feature=pd.concat((X_tfidf_feature['vocabilary'],X_tfidf_feature_filter_sum['tf_idf']),axis=1)
    X_tfidf_feature_sorted = X_tfidf_feature.sort_values(by=['tf_idf'],ascending=False)
    X_tfidf_feature_filter = X_tfidf_feature[X_tfidf_feature['tf_idf']>=50]
    print(X_tfidf_feature.shape, X_tfidf_feature_filter.shape)
    X_tfidf_feature_filter=X_tfidf_feature_filter.reset_index().rename(columns={'index':'index_feature'})
    index_features=X_tfidf_feature_filter['index_feature']

    with open(ROOT_DIR+ '/'+prefix_name+'_tfidf_filter_indexes.txt', 'w') as f:
        f.write(' '.join(index_features))
    f.close()

    X_tfidf_keywords=X_tfidf[:,index_features]

    X_tfidf_feature_filter_sorted = X_tfidf_feature_filter.sort_values(by=['index_feature'],ascending=True)
    cols_tfidf = X_tfidf_feature_filter_sorted['vocabilary'].values


    dataset_sub_features = dataset_train_test.drop(['desc', 'guid','desc_list', 'desc_list_norm', 'desc_clear_norm'],axis=1)
    dataset_sub_brand_ohe,cols_ohe = OhePreprocessing(dataset=pd.DataFrame(dataset_sub_features['brend_code']), engine='train',prefix_name=prefix_name)
    dataset_sub_features = dataset_sub_features.drop(['brend_code'],axis=1)

    cols_all = list(dataset_sub_features.columns) + cols_ohe + list(cols_tfidf)

    with open(ROOT_DIR + '/'+prefix_name+'_cols_full.csv', 'w') as f:
        f.write(str(' '.join(cols_all)))
        f.close()

    features_csr = scipy.sparse.hstack((dataset_sub_brand_ohe, X_tfidf_keywords))
    scipy.sparse.save_npz(ROOT_DIR + '/'+prefix_name+'_features_csr.npz', features_csr, compressed=True)
    dataset_sub_features.to_csv(ROOT_DIR + '/'+prefix_name+'_id_target.csv',sep=';',index=False)

if str(sys.argv[2]) =='inference':

    tfidf_vectorizer = joblib.load(ROOT_DIR+ '/'+prefix_name+'_tfidf_vectorizer.pickle')
    X_tfidf = tfidf_vectorizer.transform(dataset_train_test['desc_clear_norm'])
    X_tfidf=X_tfidf.tocsc(copy=False)

    with open(ROOT_DIR+ '/'+prefix_name+'_tfidf_filter_indexes.txt', 'r') as f:
        index_features = f.readlines()

    index_features = index_features[0]
    index_features = index_features.replace(' ','')
    index_features = index_features.replace('[','')
    index_features = index_features.replace(']','')
    index_features = index_features.split(',')

    index_features_ = [int(i) for i in index_features]

    X_tfidf_keywords=X_tfidf[:,index_features_]
    dataset_sub_features = dataset_train_test.drop(['desc', 'guid','desc_list', 'desc_list_norm', 'desc_clear_norm'],axis=1)
    dataset_sub_brand_ohe,cols_ohe = OhePreprocessing(dataset=pd.DataFrame(dataset_sub_features['brend_code']), engine='inference',prefix_name=prefix_name)
    dataset_sub_features = dataset_sub_features.drop(['brend_code'],axis=1)

    features_csr = scipy.sparse.hstack((dataset_sub_brand_ohe, X_tfidf_keywords))
    scipy.sparse.save_npz(ROOT_DIR + '/'+prefix_name+'_features_csr_inference.npz', features_csr, compressed=True)
    dataset_sub_features.to_csv(ROOT_DIR + '/'+prefix_name+'_id_target_inference.csv',sep=';',index=False)

if str(sys.argv[2]) =='inference_production':
    print('Inference production preprocessing')
    tfidf_vectorizer = joblib.load(ROOT_DIR+ '/tfidf_vectorizer.pickle')
    X_tfidf = tfidf_vectorizer.transform(dataset_train_test['desc_clear_norm'])
    X_tfidf=X_tfidf.tocsc(copy=False)

    with open(ROOT_DIR+ '/tfidf_filter_indexes.txt', 'r') as f:
        index_features = f.readlines()

    index_features = index_features[0]
    index_features = index_features.replace(' ','')
    index_features = index_features.replace('[','')
    index_features = index_features.replace(']','')
    index_features = index_features.split(',')

    index_features_ = [int(i) for i in index_features]

    X_tfidf_keywords=X_tfidf[:,index_features_]
    dataset_sub_features = dataset_train_test.drop(['desc', 'guid','desc_list', 'desc_list_norm', 'desc_clear_norm'],axis=1)
    print(dataset_sub_features['brend_code'].head(10))
    enc = joblib.load(ROOT_DIR+ '/ohe_encoder.pickle')
    dataset_sub_brand_ohe = enc.transform(dataset_sub_features['brend_code'].to_numpy().reshape(-1, 1))
    cols_ohe = []
    for col in enc.categories_[0]:
        col = 'brend_code__'+col
        cols_ohe.append(col)

   # dataset_sub_brand_ohe,cols_ohe = OhePreprocessing(dataset=dataset_sub_features['brend_code'], engine='inference_production',prefix_name=prefix_name)
    dataset_sub_features = dataset_sub_features.drop(['brend_code'],axis=1)

    features_csr = scipy.sparse.hstack((dataset_sub_brand_ohe, X_tfidf_keywords))
    scipy.sparse.save_npz(ROOT_DIR + '/'+prefix_name+'_features_csr_inference_production.npz', features_csr, compressed=True)
    dataset_sub_features.to_csv(ROOT_DIR + '/'+prefix_name+'_id_target_inference_production.csv',sep=';',index=False)
    
print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------


# ROOT_DIR = os.path.abspath(os.curdir)
# print(ROOT_DIR)
# cols = ['artical', 'brend_code', 'desc', 'guid', 'group_code']
# dataset = pd.read_csv('dataset_groups_before_analyse_sub_dataset_train_test_preprocessing_lemma.csv', dtype=object,sep='|')


# enc = joblib.load(ROOT_DIR+ '/ohe_encoder.pickle')
# print(dataset['brend_code'].head(10))
# print(dataset['brend_code'].to_numpy().reshape(-1, 1))
# cat_dummies = enc.transform(dataset['brend_code'].to_numpy().reshape(-1, 1))
# cols = []
# for col in enc.categories_[0]:
#     col = 'brend_code__'+col
#     cols.append(col)


