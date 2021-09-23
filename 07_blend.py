import warnings
warnings.simplefilter('ignore')

import os
import glob

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

s1_dir = './backup/result_data_lastday'     # lastday  0.3490
s2_dir = './backup/result_data_catboost'    # catboost 0.3106
s3_dir = './backup/result_data_prophet'     # prophet  0.3460
s4_dir = './backup/result_data_xgboost'     # xgboost  0.3146

sub_dir  = './backup/result_data'
os.makedirs(sub_dir, exist_ok=True)

for fn in tqdm(s1_files):
    real_fn = fn.split('/')[-1]
    
    tmp1_df = pd.read_csv(fn, encoding='gbk')
    tmp2_df = pd.read_csv(f'{s2_dir}/{real_fn}', encoding='gbk')
    tmp3_df = pd.read_csv(f'{s3_dir}/{real_fn}', encoding='gbk')
    tmp4_df = pd.read_csv(f'{s4_dir}/{real_fn}', encoding='gbk')
    
    sub = tmp1_df.copy()
    for col in [f'预测未来{i}小时' for i in range(1, 169)]:
        sub[col] = 0.25*(tmp1_df[col]) + 0.40*(tmp2_df[col]) + 0.25*(tmp3_df[col]) + 0.10*(tmp4_df[col])
        
    sub.to_csv(f'{sub_dir}/{real_fn}', index=False, encoding='gbk')
    
# 后处理
for fn in glob.glob(f'{sub_dir}/*.csv'):
    
    real_fn = fn.split('/')[-1].replace('.csv', '')
    category, metric, city = real_fn.split('_')
    df_sub = pd.read_csv(fn, encoding='gbk')
    print(df_sub.shape)
    df_tmp = pd.read_pickle(f'./maxmin_saved/{real_fn}.pickle')
    df_sub = pd.merge(df_sub, df_tmp, on=['UserLabel'], how='left')
    
    res = list()
    for idx, row in tqdm(df_sub.iterrows()):
        user = row['UserLabel']
        pred = row[[f'预测未来{i}小时' for i in range(1,169)]]
        if metric in ['PDCCH', 'PDSCH', 'PUSCH']:
            max_val = row['user_max']
            min_val = row['user_min']
        else:
            max_val = 1.2*row['user_max']
            min_val = 0.8*row['user_min']
        pred = np.clip(pred, min_val, max_val, out=None)
        res.append([user] + list(pred))
        
    sub = pd.DataFrame(res)
    sub.columns = ['UserLabel'] + [f'预测未来{i}小时' for i in range(1,169)]
    print(sub.shape)
    sub.to_csv(f'backup/result_data/{real_fn}.csv', index=False, encoding='gbk')
