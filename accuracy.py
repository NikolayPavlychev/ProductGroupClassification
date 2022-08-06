import os
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score,recall_score, classification_report

ROOT_DIR = os.path.abspath(os.curdir)

y_test_val = pd.read_excel(ROOT_DIR+ '/y_test_val.xlsx')
y_train_val = pd.read_excel(ROOT_DIR+ '/y_train_val.xlsx')

def target_binary(x):
    print(x)
    if (x['target']==x['predict']):
        return 1
    else:
        return 0

y_test_val['target_diff'] = y_test_val['target'] - y_test_val['predict']
ind = y_test_val[y_test_val['target_diff']==0].index
y_test_val['target_binary'] = 0
y_test_val.loc[ind,'target_binary'] = 1
ind = y_test_val[y_test_val['target_diff']!=0].index
y_test_val.loc[ind,'target_binary'] = 0
print(y_test_val['target_binary'].sum()/y_test_val.shape[0])

print(accuracy_score(y_test_val['target'],y_test_val['predict']))