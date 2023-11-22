import os
import json
import csv
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model, Bayesian_LSTM
from core.tree import ScenarioTree
from core.utils import *
from keras.utils import plot_model
from PIL import Image
import scipy.io as sio
import numpy as np
import gurobipy as grb
from gurobipy import GRB
from core.Portfolio_model_solved_v1 import *

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

#读取所需参数
configs = json.load(open('config_2.json', 'r'))
if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

#读取数据
data = DataLoader(
    os.path.join('data', configs['data']['filename']),
    configs['data']['train_test_split'],
    configs['data']['columns'],
    configs['data']['output_idx']
)

#加载训练数据
train_dataset = data.get_train_data(
    seq_len=configs['data']['sequence_length'],
    batch_size=configs['training']['batch_size'],
    normalise=configs['data']['normalise']
)

#测试数据      
x_test, y_test = data.get_test_data(             seq_len=configs['data']['sequence_length'],             normalise=configs['data']['normalise']         )   

# 加载模型
trained_model_name = "12102023-172810-e1"
trained_model = Bayesian_LSTM()
trained_model.load_model("./saved_models/"+trained_model_name+".h5")
f = trained_model.bayesian_predict(x_test,configs['data']['sequence_length'],5, False)
print(f)