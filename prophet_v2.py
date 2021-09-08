import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 500)
from tqdm.notebook import tqdm

from fbprophet import Prophet

# 配置设定

train_dir = './data'
test_dir = './compdata/4G5G_Data/提交文件样例'

sub_dir = './result_data_v2'
os.makedirs(sub_dir, exist_ok=True)

metrics = ['PDCPDL', 'PDCPUL', 'PDCCH', 'PDSCH', 'PUSCH', 'RRC']
cities = ['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']
columns_pred = ['UserLabel'] + [f'预测未来{i}小时' for i in range(1, 169)]

train_files = glob.glob(f'{train_dir}/*.pickle')
test_files = glob.glob(f'{test_dir}/*.csv')        # len 48

def run_prophet(data, metric, user):
    '''
    输入: 
        data:   user 历史时间序列 dataframe
        metric: 单个指标值
        user:   user
    输出:
        168 个未来预测值 np.array 格式
    '''
    # 获取 user 历史时间序列数据
    df = data[data['UserLabel'] == user].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 如果数据缺失, 先简单返回全零; 
    # TODO: 优化: 读取再往前的历史数据
    if len(df) == 0:
        print(f"ERR: {metric} {user}")
        return np.array([0.0] * 168, dtype=np.float32)
    
    # 按时间排序
    df = df.sort_values(by=['UserLabel', 'TimeStamp']).reset_index(drop=True)
    
    # 整理为 prophet 格式
    df = df[['TimeStamp', metric]].copy()
    df.columns = ['ds', 'y']
    df = df[df['ds'].notna()]                          # ValueError: Found NaN in column ds.
    df = df[df['y'].notna()]
    if len(df) < 12:                                   # 如果还是少于半天的数据, 放弃吧
        print(f'WARN: {metric} {user}')
        return np.array([0.0] * 168, dtype=np.float32) # TODO: 优化: 读取再往前的历史数据
    
    # 时间检查: 要预测多长的时间点
    time_delta = (pd.date_range('2021-07-07 23:00:00', periods=1, freq='H') - df['ds'].max()) / pd.Timedelta(hours=1)
    total_hours = int(time_delta.values[0])
    if total_hours < 168:                              # 虽然这不太可能会出现, 还是确保一下
        total_hours = 168

    # prophet fit
    m = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=True, 
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        uncertainty_samples=0                          # for speed up
    )
    m.fit(df)
    
    # prophet predict
    future = m.make_future_dataframe(freq='H', periods=total_hours)
    forecast = m.predict(future)
    
    # 只需要返回最后的 168 个小时
    ret = forecast['yhat'].tail(168).values
    
    return ret
  
  
def batch_run_prophet(filename):
    '''
    输入:
        filename: 48 个 submission 文件其中之一
    输出:
        符合格式的 submission 预测结果
    '''
    
    real_fn = filename.split('/')[-1].replace('.csv', '')
    category, metric, city = real_fn.split('_')            # eg. 4g, PDCPDL, C48FDFBFC4072E0E
    
    # 只需提交示例文件里的 UserLabel
    sub_df = pd.read_csv(filename, encoding='gbk')
    
    print(filename, len(sub_df))
    
    # 读取时序数据
    data = pd.read_pickle(f'{train_dir}/{category}_{metric}_{city}.pickle')
    data = data[data['TimeStamp'] > '2021-06-15'].copy()   # 只使用最近两个星期的数据
    gc.collect()
    
    # 对每一个 UserLabel 循环执行 prophet 训练及预测结果
    res = list()
    for idx, row in tqdm(sub_df.iterrows()):
        user = row['UserLabel']
        pred = run_prophet(data, metric, user)
        result = [user] + list(pred)
        res.append(result)
    
    sub_data = pd.DataFrame(res)
    sub_data.fillna(0.0, inplace=True)                      # 避免空值
    sub_data.columns = columns_pred
    
    return sub_data
  
# 多进程并发加速

def worker(test_file_list):
    for filename in test_file_list:
        sub_fn = filename.split('/')[-1]
        sub_data = batch_run_prophet(filename)
        sub_data.to_csv(f'{sub_dir}/{sub_fn}', index=False, encoding='gbk')


n_workers = 4                                #   * shit like platform *
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
        './compdata/4G5G_Data/提交文件样例/4g_RRC_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_RRC_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_RRC_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_RRC_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_EA5EAA705108BDA0.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_EA5EAA705108BDA0.csv',
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_EA5EAA705108BDA0.csv',       
    ],
    [
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_AFE97F546A10368F.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_C48FDFBFC4072E0E.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_EA5EAA705108BDA0.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPDL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDCPUL_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PDSCH_F37F452354AC87C9.csv',
        './compdata/4G5G_Data/提交文件样例/4g_PUSCH_F37F452354AC87C9.csv',
    ]
]

with Pool(n_workers) as p:
    p.map(worker, data_list)
    
    
