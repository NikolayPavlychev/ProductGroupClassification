#/ProductGroupClassification/training_inference.py created by: Nikolay Pavlychev nikolaypavlychev@yandex.ru
#-----------------------------------------------------------------------------------------------------------------------

#/home/pavlychev/anaconda3/bin/python /home/pavlychev/product_group_prediction/ProductGroupClassification/production/training_inference.py inference dataset_groups_before_analyse.csv

import time
print('Import libs')
time_start = time.time()

import os
import sys
import random
import re
import joblib
import json
import pickle
import scipy 
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import evaluate
import importlib
importlib.reload(evaluate)
from evaluate import evaluate

print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')
#-----------------------------------------------------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.curdir)

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

input_data_file = sys.argv[1]
prefix_name = input_data_file.split('.')[0]


#-----------------------------------------------------------------------------------------------------------------------
if str(sys.argv[2]) =='train':
    print('Load preprocessing data')
    time_start = time.time()

    dataset_train_test_target = pd.read_csv(ROOT_DIR + '/'+prefix_name+'_id_target.csv',sep=';')

    dataset_train_test_target = dataset_train_test_target.iloc[7:,:]
    dataset_train_test_target = dataset_train_test_target.reset_index(drop=True)
    dataset_train_test_target['target'] = dataset_train_test_target['target'].astype(int)
    y=dataset_train_test_target['target']
    print(dataset_train_test_target.shape)

    dataset_stats = pd.DataFrame(dataset_train_test_target['target'].value_counts()).reset_index()
    dataset_stats = dataset_stats.rename(columns={'index':'target','target':'count'})

    treshold = int(dataset_train_test_target.shape[0]/25000000*250)
    print('treshold=',treshold)
    dataset_stats_sub = dataset_stats[dataset_stats['count']>=treshold]

    predicted_targets = list(dataset_stats_sub['target'].values)
    print('predicted_targets size=',len(predicted_targets))

    with open(ROOT_DIR+ '/'+prefix_name+'_predicted_targets.txt', 'w') as f:
        f.write(str(predicted_targets))
        f.close()

    y_predicted = dataset_train_test_target[dataset_train_test_target['target'].isin(predicted_targets)]
    ind = list(y_predicted.index)
    target = y_predicted['target']

    classes = np.unique(y)
    features_coo = scipy.sparse.load_npz(ROOT_DIR + '/'+prefix_name+'_features_csr.npz')
    features_csr = features_coo.tocsr()[7:,:]
    print(features_csr.shape)
    features_csr = features_csr[ind,:]
    print(y_predicted.shape)
    print(features_csr.shape)
    print(target.unique().shape)

    y_predicted = y_predicted.reset_index(drop=True)
    target = target.reset_index(drop=True)

    print('Training')
    time_start = time.time()

    train_rate = features_csr.shape[0]//3*2
    X_train = features_csr[0:train_rate,:]
    y_train = target.iloc[0:train_rate]
    X_train_id = y_predicted.iloc[0:train_rate,:]
    X_test = features_csr[train_rate:,:]
    y_test = target.iloc[train_rate:]
    X_test_id = y_predicted.iloc[train_rate:,:]

    y_test = y_test.reset_index(drop=True)
    X_test_id = X_test_id.reset_index(drop=True)

    print(X_train.shape,y_train.shape,X_train_id.shape)
    print(X_test.shape,y_test.shape,X_test_id.shape)

    print(y_train.unique().shape)
    print(y_test.unique().shape)

    clf = SGDClassifier(alpha=0.0000005, loss='log', penalty='l2', n_jobs=20, shuffle=True,random_state=42,class_weight='balanced')
    clf.fit(X_train,y_train)

    joblib.dump(clf,ROOT_DIR + '/'+prefix_name+'_lr_model.pickle')

    print('Successfully!')
    print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
    print('Evaluate model')
    time_start = time.time()

    X_train_id['predict']  = clf.predict(X_train)
    print('accuracy_train=',evaluate(X_train_id,'train',str(prefix_name)))

    X_test_id['predict']  = clf.predict(X_test)
    print('accuracy_test=',evaluate(X_test_id,'test',str(prefix_name)))
        
    print('Successfully!')
    print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------

if str(sys.argv[2]) == 'inference':
    print('Load preprocessing data')
    time_start = time.time()

    dataset_train_test_target = pd.read_csv(ROOT_DIR + '/'+prefix_name+'_id_target'+'_inference.csv',sep=';')

    dataset_train_test_target = dataset_train_test_target.iloc[7:,:]
    dataset_train_test_target = dataset_train_test_target.reset_index(drop=True)
    dataset_train_test_target['target'] = dataset_train_test_target['target'].astype(int)
    y=dataset_train_test_target['target']
    print(dataset_train_test_target.shape)

    with open(ROOT_DIR+ '/'+prefix_name+'_predicted_targets.txt', 'r') as f:
        predicted_targets = f.readlines()
        f.close()

    index_target = predicted_targets[0]
    index_target = index_target.replace(' ','')
    index_target = index_target.replace('[','')
    index_target = index_target.replace(']','')
    index_target = index_target.split(',')

    index_target_ = [int(i) for i in index_target]

    y_predicted = dataset_train_test_target[dataset_train_test_target['target'].isin(index_target_)]
    ind = list(y_predicted.index)
    target = y_predicted['target']

    classes = np.unique(y)
    features_coo = scipy.sparse.load_npz(ROOT_DIR + '/'+prefix_name+'_features_csr'+'_inference.npz')
    features_csr = features_coo.tocsr()[7:,:]
    print(features_csr.shape)
    features_csr = features_csr[ind,:]
    print(y_predicted.shape)
    print(features_csr.shape)
    print(target.unique().shape)

    y_predicted = y_predicted.reset_index(drop=True)
    target = target.reset_index(drop=True)

    print('Inference')
    time_start = time.time()

    clf = joblib.load(ROOT_DIR + '/'+prefix_name+'_lr_model.pickle')
    X = features_csr
    y = target
    X_id = y_predicted

    X_id['predict']  = clf.predict(X)
    print('accuracy=',evaluate(X_id,'test',prefix_name))

    target_group_code_mapping = joblib.load(ROOT_DIR+ '/'+prefix_name+'_target_group_code_mapping_full.pickle')
    target_group_code_mapping = dict((v, int(k)) for k, v in target_group_code_mapping.items())

    X_id['group_code_predict'] = X_id['predict'].apply(lambda x: target_group_code_mapping[x])
    X_id = X_id[['artical','group_code','group_code_predict']]
    X_id.to_csv(ROOT_DIR + '/'+prefix_name+'_predictions'+'_inference.csv',sep=';',index=False)

    print('Successfully!')
    print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------

if str(sys.argv[2]) == 'inference_production':
    print('Load preprocessing data')
    time_start = time.time()

    dataset_train_test_target = pd.read_csv(ROOT_DIR + '/'+prefix_name+'_id_target'+'_inference_production.csv',sep=';')

    dataset_train_test_target = dataset_train_test_target.iloc[7:,:]
    dataset_train_test_target = dataset_train_test_target.reset_index(drop=True)
    dataset_train_test_target['target'] = dataset_train_test_target['target'].astype(int)
    y=dataset_train_test_target['target']
    print(dataset_train_test_target.shape)

    with open(ROOT_DIR+ '/predicted_targets.txt', 'r') as f:
        predicted_targets = f.readlines()
        f.close()

    index_target = predicted_targets[0]
    index_target = index_target.replace(' ','')
    index_target = index_target.replace('[','')
    index_target = index_target.replace(']','')
    index_target = index_target.split(',')

    index_target_ = [int(i) for i in index_target]

    y_predicted = dataset_train_test_target[dataset_train_test_target['target'].isin(index_target_)]
    ind = list(y_predicted.index)
    target = y_predicted['target']

    classes = np.unique(y)
    features_coo = scipy.sparse.load_npz(ROOT_DIR + '/'+prefix_name+'_features_csr'+'_inference_production.npz')
    features_csr = features_coo.tocsr()[7:,:]
    print(features_csr.shape)
    features_csr = features_csr[ind,:]
    print(y_predicted.shape)
    print(features_csr.shape)
    print(target.unique().shape)

    y_predicted = y_predicted.reset_index(drop=True)
    target = target.reset_index(drop=True)

    print('Inference')
    time_start = time.time()

    clf = joblib.load(ROOT_DIR + '/lr_prod.pickle')
    X = features_csr
    y = target
    X_id = y_predicted

    X_id['predict']  = clf.predict(X)
    print('accuracy=',evaluate(X_id,'test',prefix_name))

    target_group_code_mapping = joblib.load(ROOT_DIR+ '/target_group_code_mapping_full.pickle')
    target_group_code_mapping = dict((v, int(k)) for k, v in target_group_code_mapping.items())

    X_id['group_code_predict'] = X_id['predict'].apply(lambda x: target_group_code_mapping[x])
    X_id = X_id[['artical','group_code','group_code_predict']]
    X_id.to_csv(ROOT_DIR + '/'+prefix_name+'_predictions'+'_inference_productions.csv',sep=';',index=False)

    print('Successfully!')
    print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------