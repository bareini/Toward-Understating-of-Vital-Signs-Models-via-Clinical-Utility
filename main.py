import os
from collections import OrderedDict, defaultdict
import pickle as pkl

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from summary import utils
from summary.VSG import VSG

c_type = 'step'
imp = 'mice'
inter = 'nointer'
i = 0

base_dir = os.path.abspath(os.getcwd())

dirpath = os.path.join(base_dir, 'data')
y_test_30 = pd.read_pickle(os.path.join(dirpath, f"Y_{c_type}_30_test_{imp}_{inter}_{i}.pkl"))
X_test_30 = pd.read_pickle(os.path.join(dirpath, f"X_{c_type}_30_test_{imp}_{inter}_{i}.pkl"))

tr_dict = OrderedDict()
X_test_30['y_test_30'] = y_test_30['sys_30']
ids = X_test_30['hadm_id'].unique()

for idx in ids:
    tr_dict[idx] = utils.de_norm(X_test_30.loc[X_test_30.hadm_id == idx, 'y_test_30'].values, criteria='P_sys_q0.5')

un = {}

for k, v in tr_dict.items():
    un[k] = utils.f_norm_naive_sys(v)

trends_dict_new, res_dict = utils.get_trend(tr_dict, True)
g_norm_true = utils.get_g_norm(tr_dict, trends_dict_new)

vsg = VSG(f_norm_dict=un, g_norm_dict=g_norm_true)

vsg.calc_vsg()

tf = vsg.tf
tg = vsg.tg

with open(os.path.join(base_dir, dirpath, '30_mice.pkl'), 'rb') as file:
    sct = pkl.load(file)
ridge_test_30 = sct['30_bp_step_nointer_mice_ridge'][0]['pred_test']
xbg_test_30 = sct['30_bp_step_nointer_mice_xgb'][0]['pred_test']

pred_tr_dict_ridge = OrderedDict()
pred_tr_dict_xgb = OrderedDict()

X = X_test_30.copy()
X['ridge_test_30'] = ridge_test_30
X['xgb_test_30'] = xbg_test_30

for idx in ids:
    pred_tr_dict_ridge[idx] = X.loc[X.hadm_id == idx, 'ridge_test_30'].values
    pred_tr_dict_xgb[idx] = X.loc[X.hadm_id == idx, 'xgb_test_30'].values

outputs = {'tf': tf, 'tg': tg}

figs_loc = os.path.join(base_dir, "figs")

utils.plot_graphs(outputs, tr_dict, pred_tr_dict_ridge, label='Sys BP', pred_label='Model', figs_loc=figs_loc,
                  y_label='Systolic Blood Pressure', x_label='Time Stamp'
                  )
rand_dict = vsg.get_random_sample(8)
full_length = len(rand_dict) / 2

temp_rand_dict = defaultdict(dict)
for idx, val in rand_dict.items():
    label = 'Random'
    if idx >= full_length:
        label = 'Random2'
    index = int(idx % full_length)
    temp_rand_dict[label].update({index: val})

utils.plot_graphs(dict(temp_rand_dict), tr_dict, pred_tr_dict_ridge, label='Sys BP', pred_label='Model',
                  figs_loc=figs_loc, y_label='Systolic Blood Pressure', x_label='Time Stamp'
                  )
