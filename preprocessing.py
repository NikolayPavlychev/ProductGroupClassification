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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score,recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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
#-----------------------------------------------------------------------------------------------------------------------
print('Filtering...')
time_start = time.time()
dataset['desc'] = dataset['desc'].astype(str) 
print(dataset.loc[9,'desc'])
dataset['desc'] = parapply(dataset['desc'], lambda x: voc_filter(x))
print(dataset.loc[9,'desc'])
print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------
print('Tokenization...')
time_start = time.time()
dataset['desc_list'] = parapply(dataset['desc'], lambda x: x.split(' '))
print('process time: ',round(time.time()-time_start,2),' s')

dataset['desc_list_len'] = parapply(dataset['desc_list'], lambda x: len(x),n_jobs = 20,n_chunks=75)
# print('word count: ',dataset['desc_list_len'].sum())


print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Lemmatization...')
time_start = time.time()
dataset_sub = dataset.sample(n=100000)
from functools import lru_cache
dataset_sub['desc_list_norm'] = parapply(dataset_sub['desc_list'], lambda x: lemmatizer(x),n_jobs = 20,n_chunks=75)
print('process time: ',round(time.time()-time_start,2),' s')

time_start = time.time()
dataset_sub['desc_clear_norm'] = parapply(dataset_sub['desc_list_norm'], lambda x: ' '.join(x))

dataset_sub = dataset_sub.dropna()
print(dataset_sub.shape)
dataset_sub = dataset_sub.reset_index(drop=True)

# dataset_sub[['artical', 'brend_code', 'desc_clear_norm', 'guid', 'group_code']].to_csv(ROOT_DIR + '/dataset_groups_preprocessing_filter_v2_100000.csv',sep=';',index=False)
# dataset_sub = pd.read_csv(ROOT_DIR + '/dataset_groups_preprocessing_filter_v2_100000.csv',sep=';',dtype=object)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_tfidf = tfidf_vectorizer.fit_transform(dataset_sub['desc_clear_norm'])
X_tfidf_feature_filter_sum=X_tfidf.sum(axis=0)
X_tfidf_feature_filter_sum=pd.DataFrame(np.transpose(np.array(X_tfidf_feature_filter_sum)),columns=['tf_idf'])
X_tfidf=X_tfidf.tocsc(copy=False)


X_tfidf_feature=pd.DataFrame(tfidf_vectorizer.get_feature_names())
X_tfidf_feature=X_tfidf_feature.rename(columns={0:'vocabilary'})
X_tfidf_feature=pd.concat((X_tfidf_feature['vocabilary'],X_tfidf_feature_filter_sum['tf_idf']),axis=1)
X_tfidf_feature_sorted = X_tfidf_feature.sort_values(by=['tf_idf'],ascending=False)
# X_tfidf_feature_sorted.to_excel(ROOT_DIR+'/X_tfidf_feature_sorted_v3.xlsx')

# plt.figure(0)
# sns.distplot(X_tfidf_feature['tf_idf'],bins=1000)
# plt.show()

X_tfidf_feature_filter = X_tfidf_feature[X_tfidf_feature['tf_idf']>=2.3]
print(X_tfidf_feature.shape, X_tfidf_feature_filter.shape)
X_tfidf_feature_filter=X_tfidf_feature_filter.reset_index().rename(columns={'index':'index_feature'})
index_features=X_tfidf_feature_filter['index_feature']

X_tfidf_keywords=X_tfidf[:,index_features]
cols = []
for ind in index_features:
    cols.append(tfidf_vectorizer.get_feature_names()[ind])

dataset_sub_tfidf_pd = pd.DataFrame(data=X_tfidf_keywords.toarray(),columns=cols)

dataset_sub_features = pd.concat((dataset_sub,dataset_sub_tfidf_pd),axis=1)
#dataset_sub_features.to_csv(ROOT_DIR + '/dataset_sub_features_v3.csv',sep=';',index=False)

dataset_sub_features = pd.read_csv(ROOT_DIR + '/dataset_sub_features_v2.csv',sep=';')
# dataset_sub = pd.read_csv(ROOT_DIR + '/dataset_groups_preprocessing_filter_v2_100000.csv',sep=';',dtype=object)

dataset_sub = dataset_sub_features

dataset_sub = dataset_sub.dropna()
print(dataset_sub.shape)
dataset_sub = dataset_sub.reset_index(drop=True)

enc = LabelEncoder()
enc.fit(dataset_sub['group_code'])
dataset_sub['target'] = enc.fit_transform(dataset_sub['group_code'])

le_name_mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
print(le_name_mapping)

#import json

# le_name_mapping = json.dumps(le_name_mapping)

# with open(ROOT_DIR+ '/target_group_code_mapping.txt', 'w') as f:
#     f.write(str(le_name_mapping))


# dataset_sub_stats = pd.DataFrame(dataset_sub['target'].value_counts()).reset_index()
# dataset_sub_stats = dataset_sub_stats.rename(columns={'index':'target','target':'count'})

dataset_sub_brand_ohe = pd.get_dummies(dataset_sub['brend_code'],prefix='brand')
# print(dataset_sub.shape,dataset_sub_brand_ohe.shape,dataset_sub_features.shape)
# print(dataset_sub_brand_ohe.dtypes, dataset_sub_features.iloc[:,4:].dtypes)

dataset_train_test = pd.concat((dataset_sub[['artical','group_code']], dataset_sub_brand_ohe, 
dataset_sub.drop(['artical', 'brend_code', 'desc', 'guid', 'group_code', 'desc_list', 'desc_list_len', 'desc_list_norm', 'desc_clear_norm'],axis=1)),axis=1)
# dataset_train_test = dataset_train_test.drop(['group_code', 'desc_list', 'desc_list_len', 'desc_list_norm', 'desc_clear_norm'],axis=1)

dataset_train_test['target'] = dataset_train_test['target'].astype(int)
print(dataset_train_test.dtypes)


print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------

print('Training...')
time_start = time.time()

X_train, X_test, y_train, y_test = train_test_split(dataset_train_test.drop(['target'],axis=1),dataset_train_test['target'],test_size=0.3,random_state=42)

y_test_val = pd.DataFrame(y_test,columns=['target'])
y_test_val['artical'] = X_test['artical']
y_test_val['group_code'] = X_test['group_code']

y_train_val = pd.DataFrame(y_train,columns=['target'])
y_train_val['artical'] = X_train['artical']
y_train_val['group_code'] = X_train['group_code']

X_train = X_train.drop(['artical','group_code'],axis=1)
X_test = X_test.drop(['artical','group_code'],axis=1)


lr = LogisticRegression(C=1,random_state=42,n_jobs=-1,class_weight='balanced')
lr.fit(X_train, y_train)

print('process time: ',round(time.time()-time_start,2),' s')

print('Inference started...')

y_test_val['predict'] = lr.predict(X_test)
y_train_val['predict'] = lr.predict(X_train)

report = classification_report(y_test_val['target'], y_test_val['predict'], output_dict=True)
test_report = pd.DataFrame(report).transpose()

report = classification_report(y_train_val['target'], y_train_val['predict'], output_dict=True)
train_report = pd.DataFrame(report).transpose()

test_report = test_report.sort_values(by='f1-score',ascending=False)
train_report = train_report.sort_values(by='f1-score',ascending=False)

test_report.to_excel(ROOT_DIR+ '/test_report_vf.xlsx',index=True)
train_report.to_excel(ROOT_DIR+ '/train_report_vf.xlsx',index=True)
y_test_val.to_excel(ROOT_DIR+ '/y_test_val.xlsx',index=True)
y_train_val.to_excel(ROOT_DIR+ '/y_train_val.xlsx',index=True)

accuracy_test = accuracy_score(y_test_val['target'], y_test_val['predict'])
accuracy_train = accuracy_score(y_train_val['target'], y_train_val['predict'])
print('accuracy_test=',accuracy_test,'accuracy_train=',accuracy_train)

# balanced_accuracy_test = balanced_accuracy_score(y_test_val['target'], y_test_val['predict'])
# balanced_accuracy_train = balanced_accuracy_score(y_train_val['target'], y_train_val['predict'])
# print('balanced_accuracy_test=',balanced_accuracy_test,'balanced_accuracy_train=',balanced_accuracy_train)

print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------






