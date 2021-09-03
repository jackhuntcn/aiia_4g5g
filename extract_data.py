import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import glob
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
from tqdm.notebook import tqdm

# 配置设定

train_dir = './compdata/4G5G_Data/Train_Data'
test_dir = './compdata/4G5G_Data/提交文件样例'

cities = ['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']

# 要求提交不同个数个特定小区的 07/01 - 07/07 168 个小时的预测值

test_files = glob.glob(f'{test_dir}/*.csv')   # len 48
print(test_files)

def extract_user_data_from_file(filename):
    
    start_time = time.time()
    
    real_fn = filename.split('/')[-1].replace('.csv', '').replace('-', '_')
    category, metric, city = real_fn.split('_')
    
    # 只需提取 提交示例文件里的 UserLabel
    sub_df = pd.read_csv(filename, encoding='gbk')
    sub_users = sub_df['UserLabel'].unique()
    
    print(filename, category, metric, city, len(sub_users))
    
    res = pd.DataFrame()
    
    # 从所有的数据文件里循环获取 (暂时设定训练集为六月份)
    files =  glob.glob(f'{train_dir}/{category}*202106*{city}.csv')
    for fn in tqdm(files):
        data = pd.read_csv(fn, encoding='gbk')
        data_users = data['UserLabel'].unique()
        if len(set(sub_users) & set(data_users)) != len(sub_users):
            print(f"{fn} missing: {100 - 100*len(set(sub_users) & set(data_users)) / len(sub_users)}%")
        data = data[data['UserLabel'].isin(sub_users)].copy()
        gc.collect()
        res = pd.concat([res, data])
        del data
        gc.collect()
        
    res = res.sort_values(by=['UserLabel', 'TimeStamp']).reset_index(drop=True)
    res.to_pickle(f'prepared_data/{category}_{metric}_{city}.pickle')
    print(res.shape, 'used_time:', (time.time() - start_time) / 60, 'min')    
    
    return None
    

os.makedirs('prepared_data', exist_ok=True)

for fn in test_files:
    extract_user_data_from_file(fn)
