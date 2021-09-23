import warnings
warnings.simplefilter('ignore')

import os
import gc
import re
import glob
import time

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

data_dir = './data'

res_dir = './maxmin_saved'
os.makedirs(res_dir, exist_ok=True)

def func(data_fn):
    real_fn = data_fn.split('/')[-1].replace('.pickle', '')
    category, metric, city = real_fn.split('_')
    data = pd.read_pickle(data_fn)
    tmp = data.groupby('UserLabel')[metric].agg([('user_max', np.max), 
                                                 ('user_min', np.min),
                                                 ('user_mean', np.mean),
                                                 ('user_std', np.std)]).reset_index()
    tmp.to_pickle(f'{res_dir}/{real_fn}.pickle')
    
    
for data_fn in tqdm(glob.glob(f'{data_dir}/*.pickle')):
    func(data_fn)    
