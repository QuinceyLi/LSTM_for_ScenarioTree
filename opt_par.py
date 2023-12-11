# %%
# gpu 4060ti cpu i7
import os
import json
import csv
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model, Bayesian_LSTM
from core.tree import ScenarioTree
from core.utils import *
# from keras.utils import plot_model
from PIL import Image
import scipy.io as sio
import numpy as np
import gurobipy as grb
from gurobipy import GRB
from core.Portfolio_model_solved import *
from pylab import mpl
import seaborn as sns
from joblib import Parallel, delayed
import tensorflow as tf


# sns.set_theme(style='darkgrid') # 设置风格使图标更美观
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体雅黑，使图可以显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 忽略警告
import warnings

warnings.filterwarnings('ignore')

# 参数列表
EPOCH = [x for x in range(20, 51, 10)]
SEQ_LEN = [x for x in range(30, 61, 10)]

# 结果字典
res_dict = {}

# %%
def task(e, s):
    print(f"\n当前epoch={e},当前seq_length={s}...\n")
    print("当前进程：", os.getpid(), " 父进程：", os.getppid())
    res_dict['e' + str(e) + '_l' + str(s)] = {}
    # 读取所需参数
    configs = json.load(open('config_2.json', 'r'))
    configs['training']['epochs'] = e
    configs['data']['sequence_length'] = s
    configs['model']['layers'][0]['input_timesteps'] = s - 1

    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    # 读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['output_idx']
    )
    # 加载训练数据
    train_dataset, x_train, y_train = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise']
    )

    # 测试数据
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # print(x_train[-1,:,:])

    # %%
    # 创建模型
    BLmodel = Bayesian_LSTM()
    mymodel = BLmodel.build_model(configs)

    # %%
    # 训练模型
    stime = time.time()
    save_model_name = BLmodel.bayesian_train(
        train_dataset,
        epochs=configs['training']['epochs'],
        Num_sample=configs['training']['Num_Sample'],
        save_dir=configs['model']['save_dir'])
    # save_model_name = 'test'
    etime = time.time()
    res_dict['e' + str(e) + '_l' + str(s)]['model_train_time'] = etime - stime

    # %%
    # 生成多个情景树
    tree_model = ScenarioTree(window=x_train[-1, :, :], model=BLmodel.model, T=3, branch=[1, 15, 8, 5], n_stock=5,
                              model_name=save_model_name)
    skip = 4
    roll_step = data.len_test
    n_tree = int(roll_step / skip)

    stime = time.time()
    multi_tree, tree_name = tree_model.build_multi_trees(n_tree=n_tree, skip=skip, data=data.data_test)  # 生成多个情景树
    etime = time.time()
    res_dict['e' + str(e) + '_l' + str(s)]['tree_generate_time'] = etime - stime

    # %%
    # 读取情景树数据,做回测
    mat_dict = {
        "ARMA-GARCH": "ag_tree_04-Nov-2023 10_49_47.mat",
        "kmeans": "kmeans_tree_04-Nov-2023 13_40_44.mat",
        "moment_matching": "mm_tree_04-Nov-2023 16_49_08.mat",
        "NN": tree_name
    }

    wealth_change_dict = {
        "ARMA-GARCH": [],
        "kmeans": [],
        "moment_matching": [],
        "NN": []
    }
    decision_method = ["meancvar", "mv"]
    for d in decision_method:
        wealth_change_dict = backtesting(mat_dict, wealth_change_dict, data.data_test, d)

        w = wealth_change_dict['NN']
        w = np.array(w)
        periods = w.shape[0] * w.shape[1]

        wealth_df = pd.DataFrame(pd.date_range(start='2021-01-01', periods=periods, freq='D'), columns=['time'])
        wealth_df.set_index('time', inplace=True)

        for model_name, wealth_change_list in wealth_change_dict.items():
            wealth_change_list = np.hstack(wealth_change_list)
            wealth_df[model_name] = wealth_change_list

        # 保存数据
        res_folder = 'e' + str(e) + '_' + 'l' + str(s)

        if not os.path.exists('./results/tune_par/' + res_folder + '/'):
            os.makedirs('./results/tune_par/' + res_folder + '/')

        if not os.path.exists('./results/tune_par/' + res_folder + '/' + save_model_name + '.xlsx'):
            wealth_df.to_excel('./results/tune_par/' + res_folder + '/' + save_model_name + '.xlsx', d, index=False)
        else:
            try:
                with pd.ExcelWriter('./results/tune_par/' + res_folder + '/' + save_model_name + '.xlsx',
                                    engine='openpyxl', mode='a') as writer:
                    wealth_df.to_excel(writer, d, index=False)
            except Exception as e:
                pass

        wealth_df.plot(figsize=(12, 8), title=d)
        plt.savefig('./results/tune_par/' + res_folder + '/' + d + '.png')

if __name__ == '__main__':
    task_list = []

    for e,s in zip(EPOCH, SEQ_LEN):
        task_list.append(delayed(task)(e,s))

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    multiwork = Parallel(n_jobs=-2, backend='multiprocessing')
    res = multiwork(task_list)


    # 保存res_dict
    with open('./results/tune_par/res_dict.json', 'w') as file:
        json.dump(res_dict, file)
