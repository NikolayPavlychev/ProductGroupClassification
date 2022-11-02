import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.curdir)
cols = ['artical', 'brend_code', 'desc', 'guid', 'group_code']
dataset = pd.read_csv(ROOT_DIR + '/dataset_groups_before_analyse.csv', dtype=object,sep=';', header=0,usecols=cols)
dataset = dataset.sample(dataset.shape[0])
dataset = dataset.sample(n=100000)
dataset.to_csv(ROOT_DIR + '/dataset_groups_before_analyse_sub.csv', index=False,sep=';')

