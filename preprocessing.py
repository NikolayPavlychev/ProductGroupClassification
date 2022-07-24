#/ProductGroupClassification/preprocessing.py created by: Nikolay Pavlychev nikolaypavlychev@yandex.ru
#-----------------------------------------------------------------------------------------------------------------------
print('Import libs...')

import time
time_start = time.time()

import os
import sys
import random

import joblib
import json

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, precision_score,recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

import itertools
from itertools import product

# import OhePreprocessing
# import importlib
# importlib.reload(OhePreprocessing)
# from OhePreprocessing import OhePreprocessing

print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')

#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')
time_start = time.time()

ROOT_DIR = os.path.abspath(os.curdir)

# cols = []

dataset = pd.read_csv(ROOT_DIR + '/train_dataset_train/' + 'dataset_groups_before_analyse.csv', dtype=object,sep=',', header=0, nrows=10)

print('Successfully!')
print('process time: ',round(time.time()-time_start,2),' s')