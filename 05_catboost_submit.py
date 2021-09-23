import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import time
import glob
import joblib

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)

from tqdm.notebook import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import catboost as cbt

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
test_dir = './compdata/4G5G_Data/提交文件样例'

sub_dir = './backup/result_data_catboost'
os.makedirs(sub_dir, exist_ok=True)

metrics = ['PDCPDL', 'PDCPUL', 'PDCCH', 'PDSCH', 'PUSCH', 'RRC']
cities = ['AFE97F546A10368F', 'C48FDFBFC4072E0E', 'EA5EAA705108BDA0', 'F37F452354AC87C9']
columns_pred = ['UserLabel'] + [f'预测未来{i}小时' for i in range(1, 169)]

opt_4g = pd.read_csv('./opt_features/4g_opt.csv')
opt_5g = pd.read_csv('./opt_features/5g_opt.csv')

models_dir = './models'
os.makedirs(models_dir, exist_ok=True)

def make_feats(data, opt_data, sta_data):
    '''
    生成时间特征, 合并 opt 信息和统计特征
    '''
    # 合并 opt
    df = pd.merge(data, opt_data, on='UserLabel', how='left')
    df = df[df['TimeStamp'].notna()]
    df = df.drop_duplicates(['UserLabel', 'TimeStamp']).reset_index(drop=True)
    
    # 时间特征
    df['hour'] = df['TimeStamp'].dt.hour.astype(int)
    df['day'] = df['TimeStamp'].dt.day.astype(int)
    df['weekday'] = df['TimeStamp'].dt.weekday.astype(int)
    
    # 合并统计特征 (可能会过拟合)
    df = pd.merge(df, sta_data, on=['UserLabel', 'hour'], how='left')

    # 减少内存占用
    df = reduce_mem_usage(df)
    
    return df
  
def train_model(fn, train_epochs=100):
    '''
    输入参数为提交样例文件路径
    '''
    start_time = time.time()
    
    real_fn = fn.split('/')[-1].replace('.csv', '')
    category, metric, city = real_fn.split('_')
    
    # 读取提交文件, 并做 labelencoding
    sub_df = pd.read_csv(f'{test_dir}/{real_fn}.csv', encoding='gbk')
    print(fn, len(sub_df))
    
    lbe = LabelEncoder()
    lbe.fit(sub_df['UserLabel'])
    sub_df['UserLabel_lbe'] = lbe.transform(sub_df['UserLabel'])
    
    # 读取数据文件
    data = pd.read_pickle(f'{data_dir}/{real_fn}.pickle')
    
    # 生成特征
    opt_data = opt_4g.copy() if category == '4g' else opt_5g.copy()
    sta_data = pd.read_pickle(f'./stat_data/{real_fn}.pickle')
    df = make_feats(data, opt_data, sta_data)
    df['UserLabel_lbe'] = lbe.transform(df['UserLabel'])
    
    gc.collect()
    
    # 模型训练
    ycol = metric
    df = df[df[ycol].notna()]                 # 去掉空值
    df[ycol] = np.log1p(df[ycol])             # log1p transform
    df = df[df[ycol].notna()]                 # 再次去掉空值
    
    not_used = ['UserLabel', 'TimeStamp', 'city'] + ['longitude', 'latitude', 'euclidean']
    if category == '5g':
        not_used = not_used + ['band', 'freq_num', 'freq_band', 'freq_point']
        
    feature_names = list(
        filter(lambda x: x not in [ycol] + not_used, df.columns))
    X_train = df[feature_names]                  
    Y_train = df[ycol].astype(float)
    
    model = cbt.CatBoostRegressor(task_type='GPU',
                                  learning_rate=0.1,
                                  loss_function='RMSE',
                                  iterations=train_epochs,
                                  random_seed=42,
                                  max_depth=5,
                                  reg_lambda=0.5,
                                  early_stopping_rounds=100)

    cbt_model = model.fit(X_train,
                          Y_train,
                          eval_set=[(X_train, Y_train)],
                          verbose=100,
                          early_stopping_rounds=100)
    
    # 保存
    joblib.dump(cbt_model, f'./models/cbt_model_{real_fn}.pickle')
    print('used time:', time.time() - start_time, 'seconds')
    
    return cbt_model, feature_names, lbe  
  
def batch_predict(fn, lbe, model, feature_names):
    '''
    输入参数为提交样例文件路径
    '''
    start_time = time.time()
    
    real_fn = fn.split('/')[-1].replace('.csv', '')
    category, metric, city = real_fn.split('_')
    sub_df = pd.read_csv(f'{test_dir}/{real_fn}.csv', encoding='gbk')
    opt_data = opt_4g.copy() if category == '4g' else opt_5g.copy()
    sta_data = pd.read_pickle(f'./stat_data/{real_fn}.pickle')
    print(fn, len(sub_df))
    
    df_tpl = pd.DataFrame({'TimeStamp': pd.date_range(start='2021-07-01 00:00:00', freq='1H', periods=168)})
    df_tpl['hour'] = df_tpl['TimeStamp'].dt.hour.astype(int)
    df_tpl['day'] = df_tpl['TimeStamp'].dt.day.astype(int)
    df_tpl['weekday'] = df_tpl['TimeStamp'].dt.weekday.astype(int)
    
    chuck_size = 1000
    total_iter = len(sub_df) // chuck_size + 1     # 这些数据集里没有整除的, 先这样简单处理了
    
    sub_all = pd.DataFrame()
    
    for it in tqdm(range(total_iter)):
        iter_df = sub_df[it*chuck_size:(it+1)*chuck_size].copy()
    
        test_data = pd.DataFrame()
        for idx, row in iter_df.iterrows():
            user = row['UserLabel']
            tmp = df_tpl.copy()
            tmp['UserLabel'] = user
            test_data = pd.concat([test_data, tmp])
    
        test_data = test_data.merge(opt_data, on='UserLabel', how='left')
        test_data = test_data.merge(sta_data, on=['UserLabel', 'hour'], how='left')
        test_data['UserLabel_lbe'] = lbe.transform(test_data['UserLabel'])
    
        pred = model.predict(test_data[feature_names])
        pred = np.expm1(pred)               # inv log1p transform
        test_data['pred'] = pred 
        sub_data = test_data[['UserLabel', 'TimeStamp', 'pred']].copy()
        sub_data = pd.DataFrame(sub_data.groupby(['UserLabel'])['pred'].agg(list))
        sub = pd.DataFrame({'UserLabel': sub_data.index})
        for i in range(168):
            sub[f'预测未来{i+1}小时'] = sub_data['pred'].apply(lambda x: x[i]).values
            
        sub_all = pd.concat([sub_all, sub])
    
    # 检查
    test_df = pd.read_csv(fn, encoding='gbk')
    print("CHECK:", any(test_df['UserLabel'].values == sub_all['UserLabel'].values))
    
    print('used time:', time.time() - start_time, 'seconds')
    return sub_all  
  
  
test_files = glob.glob(f'{test_dir}/*.csv')

for idx, fn in enumerate(test_files):
    print('running:', idx+1, fn)    
    real_fn = fn.split('/')[-1].replace('.csv', '')
    
    cbt_model, feature_names, lbe = train_model(fn, train_epochs=3000)
    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': cbt_model.feature_importances_,
    })
    print(df_importance.sort_values('importance', ascending=False))
    sub = batch_predict(fn, lbe, cbt_model, feature_names)
    sub.to_csv(f'{sub_dir}/{real_fn}.csv', index=False, encoding='gbk')  
    

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
