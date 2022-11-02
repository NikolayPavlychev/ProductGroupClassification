#/ProductGroupClassification/training_evaluate_model.py created by: Nikolay Pavlychev nikolaypavlychev@yandex.ru
#-----------------------------------------------------------------------------------------------------------------------
#Calculate accuracy per group

import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.abspath(os.curdir)

def evaluate(X_id,label,prefix_name):
    accuracy = accuracy_score(X_id['target'], X_id['predict'])

    X_id['target_diff'] = X_id['target'] - X_id['predict']
    ind = X_id[X_id['target_diff']==0].index
    X_id['target_binary'] = 0
    X_id.loc[ind,'target_binary'] = 1
    ind = X_id[X_id['target_diff']!=0].index
    X_id.loc[ind,'target_binary'] = 0

    accuracy_per_group_sum = X_id.groupby(by=['target'])['target_binary'].apply(np.sum)
    accuracy_per_group_count = X_id.groupby(by=['target'])['target_binary'].apply(lambda x: x.shape[0])
    accuracy_per_group_sum = accuracy_per_group_sum.reset_index().rename(columns={'target':'label','target_binary':'sum'})
    accuracy_per_group_count = accuracy_per_group_count.reset_index().rename(columns={'target':'label','target_binary':'count'})
    accuracy_per_group = accuracy_per_group_sum.merge(accuracy_per_group_count,on=['label'],how='inner')
    accuracy_per_group['accuracy'] = accuracy_per_group['sum']/accuracy_per_group['count']
    accuracy_per_group.sort_values(by=['accuracy'],ascending=False,inplace=True)
    print(ROOT_DIR+'/'+prefix_name+'_y_'+label+'_accuracy.xlsx')
    accuracy_per_group.to_excel(ROOT_DIR+ '/'+prefix_name+'_y_'+label+'_accuracy.xlsx')

    return accuracy
        
#-----------------------------------------------------------------------------------------------------------------------