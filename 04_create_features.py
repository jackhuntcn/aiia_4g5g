import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import glob
import time

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
  
  
data_dir = './data'

## OPT features

opt_4g = pd.read_csv('compdata/4G5G_Data/Opt_Data/4g_cfg.csv', encoding='gbk')
opt_5g = pd.read_csv('compdata/4G5G_Data/Opt_Data/5g_cfg_nrcell.csv', encoding='gbk')

print(opt_4g.shape, opt_5g.shape)

sub_4g_files = glob.glob('./compdata/4G5G_Data/提交文件样例/4g*.csv')
sub_5g_files = glob.glob('./compdata/4G5G_Data/提交文件样例/5g*.csv')

users_4g = list()
for fn in tqdm(sub_4g_files):
    df = pd.read_csv(fn, encoding='gbk')
    users = list(df['UserLabel'].unique())
    users_4g.extend(users)
    
    
users_5g = list()
for fn in tqdm(sub_5g_files):
    df = pd.read_csv(fn, encoding='gbk')
    users = list(df['UserLabel'].unique())
    users_5g.extend(users)
    
    
users_4g = list(set(users_4g))
users_5g = list(set(users_5g))

print(len(users_4g), len(users_5g))

print(opt_4g.shape, opt_5g.shape)
opt_4g = opt_4g[opt_4g['小区中文名'].isin(users_4g)]
opt_5g = opt_5g[opt_5g['小区中文名'].isin(users_5g)]
print(opt_4g.shape, opt_5g.shape)

opt_4g = opt_4g.drop_duplicates().reset_index(drop=True)
opt_5g = opt_5g.drop_duplicates().reset_index(drop=True)

print(opt_4g.shape, opt_5g.shape)

for col in ['电子下倾角', '机械下倾角', '总下倾角', '方位角', '天线挂高']:
    opt_4g[col] = opt_4g.groupby('小区中文名')[col].transform('mean')
    opt_5g[col] = opt_5g.groupby('小区中文名')[col].transform('mean')
    
opt_4g = opt_4g.drop_duplicates(['小区中文名'], keep='first').reset_index(drop=True)
opt_5g = opt_5g.drop_duplicates(['小区中文名'], keep='first').reset_index(drop=True)

print(opt_4g.shape, opt_5g.shape)

for col in ['经度', '纬度']:
    opt_4g[col].fillna(0, inplace=True)
    opt_5g[col].fillna(0, inplace=True)

opt_4g['euclidean'] = (opt_4g['经度']**2 + opt_4g['纬度']**2)**0.5
opt_5g['euclidean'] = (opt_5g['经度']**2 + opt_5g['纬度']**2)**0.5

id_cols  = ['地市', '小区中文名']
cat_cols = ['STATE', '制式', '覆盖类型', '覆盖场景', '工作频段', '中心载频的信道号', '频段指示']
num_cols = ['跟踪区码', '物理小区识别码', '中心频点', '带宽', '总下倾角', '方位角', '天线挂高', '最大发射功率', '经度', '纬度', 'euclidean']

for col in cat_cols:
    opt_4g[col].fillna('_NaN_', inplace=True)
    opt_5g[col].fillna('_NaN_', inplace=True)
    
for col in num_cols:
    opt_4g[col].fillna(-1, inplace=True)
    opt_5g[col].fillna(-1, inplace=True)
    
for col in cat_cols:
    lbe = LabelEncoder()
    opt_4g[col] = opt_4g[col].astype(str)
    opt_4g[col] = lbe.fit_transform(opt_4g[col])
    
for col in cat_cols:
    lbe = LabelEncoder()
    opt_5g[col] = opt_5g[col].astype(str)
    opt_5g[col] = lbe.fit_transform(opt_5g[col])
    
opt_4g = opt_4g[id_cols + cat_cols + num_cols]
opt_5g = opt_5g[id_cols + cat_cols + num_cols]

en_cols = ['city', 'UserLabel', 
           'state', 'system', 'coverage_type', 'coverage_scene', 'freq_band', 'freq_num', 'band',
           'lac', 'ci', 'freq_point', 'band_width', 'total_tilt', 'direction', 'height', 'max_power', 'longitude', 'latitude', 'euclidean']
opt_4g.columns = en_cols
opt_5g.columns = en_cols

cities_mapping = {
    'AFE97F546A10368F': 1,
    'C48FDFBFC4072E0E': 2,
    'EA5EAA705108BDA0': 3, 
    'F37F452354AC87C9': 4
}

opt_4g['city'] = opt_4g['city'].map(cities_mapping)
opt_4g.head()

opt_5g['city'] = opt_5g['city'].map(cities_mapping)
opt_5g.head()

save_dir = './opt_features'
os.makedirs(save_dir, exist_ok=True)

os.makedirs(f'{save_dir}/state', exist_ok=True)
os.makedirs(f'{save_dir}/coverage_scene', exist_ok=True)

def func1(data_fn, to_save=True):
    
    real_fn = data_fn.split('/')[-1].replace('.pickle', '')
    category, metric, city = real_fn.split('_')
    
    # 读取数据并做预处理
    data = pd.read_pickle(data_fn)
    data = data.drop_duplicates(['UserLabel', 'TimeStamp']).reset_index(drop=True)

    # 合并 opt 数据
    opt_data = opt_4g.copy() if category == '4g' else opt_5g.copy()
    data = pd.merge(data, opt_data, on='UserLabel', how='left')

    # state 目标编码
    tmp1 = data.groupby('state')[metric].agg([('mean_state', np.mean),
                                              ('max_state', np.max),
                                              ('std_state', np.std)]).reset_index()
    tmp1['state'] = tmp1['state'].astype(int)
    
    # coverage_scene 目标编码
    tmp2 = data.groupby('coverage_scene')[metric].agg([('mean_coverage_scene', np.mean),
                                                       ('max_coverage_scene', np.max),
                                                       ('std_coverage_scene', np.std)]).reset_index()
    tmp2['coverage_scene'] = tmp2['coverage_scene'].astype(int)
    
    # 保存
    if to_save:
        tmp1.to_pickle(f'{save_dir}/state/{real_fn}.pickle')
        tmp2.to_pickle(f'{save_dir}/coverage_scene/{real_fn}.pickle')
        
    return tmp1, tmp2
  
for data_fn in tqdm(glob.glob(f'{data_dir}/*.pickle')):
    func1(data_fn)
    
opt_4g.to_csv(f'{save_dir}/4g_opt.csv', index=False)
opt_5g.to_csv(f'{save_dir}/5g_opt.csv', index=False)


## Series features

save_dir = './series_features'
os.makedirs(save_dir, exist_ok=True)

os.makedirs(f'{save_dir}/hour', exist_ok=True)
os.makedirs(f'{save_dir}/weekday', exist_ok=True)

def func2(data_fn, to_save=True):
    
    real_fn = data_fn.split('/')[-1].replace('.pickle', '')
    category, metric, city = real_fn.split('_')
    
    # 读取数据并做预处理
    data = pd.read_pickle(data_fn)
    data = data.drop_duplicates(['UserLabel', 'TimeStamp']).reset_index(drop=True)
    data['day'] = data['TimeStamp'].dt.day
    data['hour'] = data['TimeStamp'].dt.hour
    data['weekday'] = data['TimeStamp'].dt.weekday
    
    # user 目标编码
    tmp1 = data.groupby('UserLabel')[metric].agg([('user_max', np.max), 
                                                  ('user_min', np.min),
                                                  ('user_mean', np.mean),
                                                  ('user_std', np.std)]).reset_index()
    # day 目标编码 汇总
    data['day_max'] = data.groupby(['UserLabel', 'day'])[metric].transform('max')
    data['day_min'] = data.groupby(['UserLabel', 'day'])[metric].transform('min')
    tmp2 = data.groupby(['UserLabel'])['day_max'].agg([('mean_day_max', np.mean),
                                                       ('std_day_max', np.std)]).reset_index()
    tmp3 = data.groupby(['UserLabel'])['day_min'].agg([('mean_day_min', np.mean),
                                                       ('std_day_min', np.std)]).reset_index()
    # user-hour 目标编码 
    tmp4 = data.groupby(['UserLabel', 'hour'])[metric].agg([('mean_hour', np.mean),
                                                            ('max_hour', np.max),
                                                            ('min_hour', np.min),
                                                            ('std_hour', np.std)]).reset_index()
    # user-hour 变化量目标编码
    data['shift1'] = data.groupby('UserLabel')[metric].shift(1)
    data['diff1'] = data['shift1'] - data[metric]
    tmp5 = data.groupby(['UserLabel', 'hour'])['diff1'].agg([('mean_hour_diff1', np.mean),
                                                             ('max_hour_diff1', np.max),
                                                             ('min_hour_diff1', np.min),
                                                             ('std_hour_diff1', np.std)]).reset_index()
    
    # weekday 目标编码
    tmp6 = data.groupby(['UserLabel', 'weekday'])[metric].agg([('mean_weekday', np.mean),
                                                               ('max_weekday', np.max),
                                                               ('min_weekday', np.min),
                                                               ('std_weekday', np.std)]).reset_index()
    tmp6['weekday'] = tmp6['weekday'].astype(int)
    
    # 合并
    tmp = pd.merge(tmp4, tmp5, on=['UserLabel', 'hour'], how='left')
    tmp = pd.merge(tmp, tmp1, on='UserLabel', how='left')
    tmp = pd.merge(tmp, tmp2, on='UserLabel', how='left')
    tmp = pd.merge(tmp, tmp3, on='UserLabel', how='left')
    tmp['hour'] = tmp['hour'].astype(int)
    
    # 手动计算差值
    tmp['mean_diff'] = tmp['mean_day_max'] - tmp['mean_day_min']
    
    if to_save:
        tmp.to_pickle(f'{save_dir}/hour/{real_fn}.pickle')
        tmp6.to_pickle(f'{save_dir}/weekday/{real_fn}.pickle')
        
    return tmp, tmp6
  
for data_fn in tqdm(glob.glob(f'{data_dir}/*.pickle')):
    func2(data_fn)
    
    
