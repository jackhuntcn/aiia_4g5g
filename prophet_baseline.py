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

train_dir = './prepared_data'
test_dir = './compdata/4G5G_Data/提交文件样例'

sub_dir = './result_data'
os.makedirs(sub_dir, exist_ok=True)

metrics = ['PDCPDL', 'PDCPUL', 'PDCCH', 'PDSCH', 'PUSCH', 'RRC']
cities = ['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']
columns_common = ['city', 'TimeStamp', 'VendorName', 'UserLabel']
columns_4g_mapping = {
    '上行利用率PUSCH': 'PUSCH',
    '下行利用率PDSCH': 'PDSCH',
    '下行利用率PDCCH': 'PDCCH',
    '有效RRC连接平均数': 'RRC',
    '上行流量': 'PDCPUL',
    '下行流量': 'PDCPDL'
}
columns_5g_mapping = {
    '上行利用率PUSCH': 'PUSCH',
    '下行利用率PDSCH': 'PDSCH',
    '下行利用率PDCCH': 'PDCCH',
    '有数据传输的RRC数': 'RRC',
    '上行流量': 'PDCPUL',
    '下行流量': 'PDCPDL'
}
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
        return np.array([0.0] * 168, dtype=np.float32)
    
    # 按时间排序
    df = df.sort_values(by=['UserLabel', 'TimeStamp']).reset_index(drop=True)
    
    # 整理为 prophet 格式
    df = df[['TimeStamp', metric]].copy()
    df.columns = ['ds', 'y']
    df = df[df['ds'].notna()]                          # ValueError: Found NaN in column ds.
    df = df[df['y'].notna()]
    if len(df) < 12:                                   # 如果还是少于半天的数据, 还是放弃吧
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
        uncertainty_samples=0                          # for speed up
    )
    m.fit(df)
    
    # prophet predict
    future = m.make_future_dataframe(freq='H', periods=total_hours)
    forecast = m.predict(future)
    
    # 只需要返回最后的 168 个小时
    return forecast['yhat'].tail(168).values
    
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
    data = pd.read_pickle(f'prepared_data/{category}_{metric}_{city}.pickle')
    columns_mapping = columns_4g_mapping if category == '4g' else columns_5g_mapping
    data = data.rename(columns=columns_mapping)
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
    data = data[data['TimeStamp'] > '2021-06-15'].copy()                    # 只使用最近两个星期的数据
    data = data[['UserLabel', 'TimeStamp', metric]].copy()
    gc.collect()
    
    # 对每一个 UserLabel 循环执行 prophet 训练及预测结果
    res = list()
    for idx, row in tqdm(sub_df.iterrows()):
        user = row['UserLabel']
        pred = run_prophet(data, metric, user)
        result = [user] + list(pred)
        res.append(result)
    
    sub_data = pd.DataFrame(res)
    sub_data.columns = columns_pred
    
    return sub_data
    
# 多进程并发加速

def worker(test_file_list):
    for filename in test_file_list:
        sub_fn = filename.split('/')[-1]
        sub_data = batch_run_prophet(filename)
        sub_data.to_csv(f'{sub_dir}/{sub_fn}', index=False, encoding='gbk')


# 以下随机划分并不太合理, 会出现将耗时比较长的 4g 文件放在一个队列里, 所以先手动按城市划分 
# n_workers = 8
# chuck_size = 6   # 48/8
# data_list = [test_files[:chuck_size]] + \
#     [test_files[i*(chuck_size):(i+1)*(chuck_size)] for i in range(1, n_workers-1)] + \
#     [test_files[(n_workers-1)*chuck_size:]]


n_workers = 4                                            #   * shit like platform *
data_list = [
    [f for f in test_files if re.search(cities[0], f)],
    [f for f in test_files if re.search(cities[1], f)],
    [f for f in test_files if re.search(cities[2], f)],
    [f for f in test_files if re.search(cities[3], f)],
]

with Pool(n_workers) as p:
    p.map(worker, data_list)
    
