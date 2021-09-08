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
pd.set_option('float_format', lambda x: '%.6f' % x)
from tqdm.notebook import tqdm

sub_dir = './backup/result_data'
sub_files = glob.glob(f'{sub_dir}/*.csv')

test_dir = './compdata/4G5G_Data/提交文件样例'
test_files = glob.glob(f'{test_dir}/5g*.csv')

columns_pred = ['UserLabel'] + [f'预测未来{i}小时' for i in range(1, 169)]

# 后处理 1: 把负数都变成 0
for fn in tqdm(sub_files):
    tmp = pd.read_csv(fn, encoding='gbk')
    for col in [f'预测未来{i}小时' for i in range(1, 169)]:
        tmp.loc[tmp[col] < 0, col] = 0
    tmp.to_csv(fn, index=False, encoding='gbk')
    
# 后处理 2: 把 PUSCH, PDSCH, PDCCH 三个利用率指标的最大值限制为 100
for fn in tqdm(sub_files):
    if re.search('PUSCH|PDSCH|PDCCH', fn):
        print(fn)
        tmp = pd.read_csv(fn, encoding='gbk')
        for col in [f'预测未来{i}小时' for i in range(1, 169)]:
            tmp.loc[tmp[col] > 100, col] = 100
        tmp.to_csv(fn, index=False, encoding='gbk')
        
# 掉分
# 后处理 3: PDCPUL, PDCPDL, RRC 最大值截断

# cutoff_dict = {
#     '4g': {
#         'AFE97F546A10368F': {
#             'RRC': 700,
#             'PDCPUL': 20000000,
#             'PDCPDL': 45000000,
#         },
#         'C48FDFBFC4072E0E': {
#             'RRC': 500,
#             'PDCPUL': 21000000,
#             'PDCPDL': 49000000,
#         },
#         'EA5EAA705108BDA0': {
#             'RRC': 800,
#             'PDCPUL': 16000000,
#             'PDCPDL': 54000000,
#         },
#         'F37F452354AC87C9': {
#             'RRC': 300,
#             'PDCPUL': 12000000,
#             'PDCPDL': 31000000,
#         }
#     },
#     '5g': {
#         'AFE97F546A10368F': {
#             'RRC': 400,
#             'PDCPUL':  50000000,
#             'PDCPDL': 250000000,
#         },
#         'C48FDFBFC4072E0E': {
#             'RRC': 200,
#             'PDCPUL':  92000000,
#             'PDCPDL': 990000000,
#         },
#         'EA5EAA705108BDA0': {
#             'RRC': 300,
#             'PDCPUL': 20000000,
#             'PDCPDL': 61000000,
#         },
#         'F37F452354AC87C9': {
#             'RRC': 200,
#             'PDCPUL':  12000000,
#             'PDCPDL': 320000000,
#         }
#     }
# }

for cate in ['4g', '5g']:
    for city in tqdm(['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']):
#         for metric in ['PDCPDL', 'PDCPUL', 'RRC']:
        for metric in ['RRC']:
            fn = f'{sub_dir}/{cate}_{metric}_{city}.csv'
            max_ = 400 if cate == '5g' else 800 # cutoff_dict[cate][city][metric]
            print(fn, max_)
            tmp = pd.read_csv(fn, encoding='gbk')
            for col in [f'预测未来{i}小时' for i in range(1, 169)]:
                tmp.loc[tmp[col] > max_, col] = max_
            tmp.to_csv(fn, index=False, encoding='gbk')
            
            
