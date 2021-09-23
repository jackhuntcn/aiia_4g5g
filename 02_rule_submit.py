import warnings
warnings.simplefilter('ignore')

import os
import re
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)

from tqdm.notebook import tqdm

train_dir = './data'
test_dir = './compdata/4G5G_Data/提交文件样例'

sub_dir = './result_data_lastday'
os.makedirs(sub_dir, exist_ok=True)

metrics = ['PDCPDL', 'PDCPUL', 'PDCCH', 'PDSCH', 'PUSCH', 'RRC']
cities = ['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']
columns_pred = ['UserLabel'] + [f'预测未来{i}小时' for i in range(1, 169)]

test_files = glob.glob(f'{test_dir}/*.csv')

def rule_fill(data, metric, user):
    '''
    找到该 user 的最后 24 个数据进行填充
    这里的 data 只有最后一天 (2021-06-30) 的数据
    '''
    tmp_df = data[data['UserLabel'] == user].copy()
    tmp_df = tmp_df.drop_duplicates().reset_index(drop=True)
    
    if len(tmp_df) == 0:                             # 最后一天没有数据的情况
        print(f"WARN1: {metric} {user}")
        return np.array([0.0]*24, dtype=np.float32)
    
    tmp_df = tmp_df.sort_values(
        by=['TimeStamp']).reset_index(drop=True)     # 排序确保不会出错
    
    if len(tmp_df) < 24:
        print(f"WARN2: {metric} {user}")
        df = pd.DataFrame({'TimeStamp': pd.date_range(start='2021-06-30 00:00:00', freq='1H', periods=24)})
        df = df.merge(tmp_df, on='TimeStamp', how='left')
        df[metric].fillna(method='ffill', inplace=True)
        return df[metric].values
    
    if len(tmp_df) > 24:
        print(f"WARN3: {metric} {user}")
        return tmp_df.tail(24)[metric].values
    
    return tmp_df[metric].values
  
  
def batch_run(filename):
    '''
    参数为提交文件的路径
    '''
    sub_df = pd.read_csv(filename, encoding='gbk')
    
    print(filename, len(sub_df))
    
    real_fn = filename.split('/')[-1].replace('.csv', '')
    category, metric, city = real_fn.split('_') 
    
    data = pd.read_pickle(f'{train_dir}/{real_fn}.pickle')
    data = data[data['TimeStamp'] >= '2021-06-30'].copy()
    
    res = list()
    for idx, row in tqdm(sub_df.iterrows()):
        user = row['UserLabel']
        pred = rule_fill(data, metric, user)
        result = [user] + list(pred) + list(pred) + list(pred) +\
                          list(pred) + list(pred) + list(pred) +\
                          list(pred)
        res.append(result)
    
    sub_data = pd.DataFrame(res)
    sub_data.columns = columns_pred
    
    return sub_data
  
  
# 多进程并发加速

def worker(test_file_list):
    for filename in test_file_list:
        sub_fn = filename.split('/')[-1]
        sub_data = batch_run(filename)
        sub_data.to_csv(f'{sub_dir}/{sub_fn}', index=False, encoding='gbk')


n_workers = 8                                #   * shit like platform *
data_list = [
    [
        './compdata/4G5G_Data/提交文件样例/5g_PDCCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPDL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPDL_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPDL_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPDL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPUL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPUL_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPUL_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDCPUL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDSCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDSCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PDSCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PUSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PUSCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PUSCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_PUSCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/5g_RRC_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/5g_RRC_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/5g_RRC_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/5g_RRC_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_RRC_F37F452354AC87C9.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_EA5EAA705108BDA0.csv', 
        './compdata/4G5G_Data/提交文件样例/4g_RRC_EA5EAA705108BDA0.csv',
        
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_RRC_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_RRC_C48FDFBFC4072E0E.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_C48FDFBFC4072E0E.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_C48FDFBFC4072E0E.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_C48FDFBFC4072E0E.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_C48FDFBFC4072E0E.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_C48FDFBFC4072E0E.csv',
    ]
]

with Pool(n_workers) as p:
    p.map(worker, data_list)  
