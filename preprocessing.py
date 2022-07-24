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
import pymorphy2

import parapply
import importlib
importlib.reload(parapply)
from parapply import parapply

import itertools
from itertools import product

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

print('Lemmatization...')
dataset['desc_list'] = parapply(dataset['desc_list'], lambda x: lemmatizer(x),n_jobs = 20,n_chunks=10)
dataset['desc_clear_norm'] = dataset['desc_list']
print('process time: ',round(time.time()-time_start,2),' s')







