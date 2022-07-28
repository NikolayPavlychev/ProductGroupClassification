#/ProductGroupClassification/preprocessing.py created by: Nikolay Pavlychev nikolaypavlychev@yandex.ru
#-----------------------------------------------------------------------------------------------------------------------
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
from sklearn.metrics import recall_score, precision_score,recall_score, classification_report
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



import functions
import importlib
importlib.reload(functions)
from functions import *




print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')
time_start = time.time()

ROOT_DIR = os.path.abspath(os.curdir)

cols = ['artical', 'brend_code', 'desc', 'guid', 'group_code']

dataset = pd.read_csv(ROOT_DIR + '/dataset_groups_before_analyse.csv', dtype=object,sep=';', header=0,usecols=cols)
print('dataset shape: ',dataset.shape)
print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')

print('Filtering...')
time_start = time.time()
dataset['desc'] = dataset['desc'].astype(str) 
print(dataset.loc[8,'desc'])
dataset['desc'] = parapply(dataset['desc'], lambda x: voc_filter(x))
print(dataset.loc[8,'desc'])
print('process time: ',round(time.time()-time_start,2),' s')

print('Tokenization...')
time_start = time.time()
dataset['desc_list'] = parapply(dataset['desc'], lambda x: x.split(' '))
print('process time: ',round(time.time()-time_start,2),' s')

dataset['desc_list_len'] = parapply(dataset['desc_list'], lambda x: len(x),n_jobs = 20,n_chunks=75)
print('word count: ',dataset['desc_list_len'].sum())


# doc = Doc(dataset.loc[0,'desc'])
# segmenter = Segmenter()
# morph_vocab = MorphVocab()
# doc.segment(segmenter)
# for token in doc.tokens:
#     token.lemmatize(morph_vocab)
# {_.lemma for _ in doc.tokens}

print('Lemmatization...')
time_start = time.time()
dataset_sub = dataset.sample(n=100000)
from functools import lru_cache
dataset_sub['desc_list_norm'] = parapply(dataset_sub['desc_list'], lambda x: lemmatizer(x),n_jobs = 20,n_chunks=75)
print('process time: ',round(time.time()-time_start,2),' s')

time_start = time.time()
dataset_sub['desc_clear_norm'] = parapply(dataset_sub['desc_list_norm'], lambda x: ' '.join(x))
# dataset_sub[['artical', 'brend_code', 'desc_clear_norm', 'guid', 'group_code']].to_csv(ROOT_DIR + '/dataset_groups_preprocessing_100000.csv',sep=';',index=False)

dataset_sub = pd.read_csv(ROOT_DIR + '/dataset_groups_preprocessing_100000.csv',sep=';',dtype=object)

dataset_sub = dataset_sub.dropna()
print(dataset_sub.shape)
dataset_sub = dataset_sub.reset_index(drop=True)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_tfidf = tfidf_vectorizer.fit_transform(dataset_sub['desc_clear_norm'])
X_tfidf_feature_filter_sum=X_tfidf.sum(axis=0)
X_tfidf_feature_filter_sum=pd.DataFrame(np.transpose(np.array(X_tfidf_feature_filter_sum)),columns=['tf_idf'])
X_tfidf=X_tfidf.tocsc(copy=False)
#scipy.sparse.save_npz(ROOT_DIR+'/X_tfidf_subset.npz')
# X_tfidf=scipy.sparse.load_npz(ROOT_DIR+'/X_tfidf_subset.npz')

X_tfidf_feature=pd.DataFrame(tfidf_vectorizer.get_feature_names())
X_tfidf_feature=X_tfidf_feature.rename(columns={0:'vocabilary'})
X_tfidf_feature=pd.concat((X_tfidf_feature['vocabilary'],X_tfidf_feature_filter_sum['tf_idf']),axis=1)


plt.figure(0)
sns.distplot(X_tfidf_feature['tf_idf'],bins=100)
plt.show()

X_tfidf_feature_filter = X_tfidf_feature[X_tfidf_feature['tf_idf']>=21]
print(X_tfidf_feature.shape, X_tfidf_feature_filter.shape)
X_tfidf_feature_filter=X_tfidf_feature_filter.reset_index().rename(columns={'index':'index_feature'})
index_features=X_tfidf_feature_filter['index_feature']

X_tfidf_keywords=X_tfidf[:,index_features]
cols = []
for ind in index_features:
    cols.append(tfidf_vectorizer.get_feature_names()[ind])

dataset_sub_tfidf_pd = pd.DataFrame(data=X_tfidf_keywords.toarray(),columns=cols)

dataset_sub_features = pd.concat((dataset_sub,dataset_sub_tfidf_pd),axis=1)
dataset_sub_features.to_csv(ROOT_DIR + '/dataset_sub_features.csv',sep=';',index=False)

print('process time: ',round(time.time()-time_start,2),' s')

enc = LabelEncoder()
dataset_sub['target'] = enc.fit_transform(dataset_sub['group_code'])
dataset_sub_stats = pd.DataFrame(dataset_sub['target'].value_counts()).reset_index()











