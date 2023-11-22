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


def main():
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
    #创建RNN模型
    BLmodel = Bayesian_LSTM()
    mymodel = BLmodel.build_model(configs)
    
    plot_model(mymodel, to_file='./pic/model.png',show_shapes=True)
    
    #加载训练数据
    train_dataset = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise']
    )
    # print(train_dataset)
    
	#训练模型
    BLmodel.bayesian_train(train_dataset, epochs=configs['training']['epochs'],Num_sample=configs['training']['Num_Sample'],
                         save_dir=configs['model']['save_dir'])
    
    #测试结果         
    x_test, y_test = data.get_test_data(             seq_len=configs['data']['sequence_length'],             normalise=configs['data']['normalise']         )         
    # print("x_test:shape",x_test.shape, "y_test:shape",y_test.shape)
    	
    is_Plot = False
    # 通过绘制stock1的预测情况，可视化模型的效果
    if is_Plot:
        #展示测试效果
        predictions_multiseq = BLmodel.bayesian_predict(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'], plot=True)       

        plot_results_multiple(predictions_multiseq, y_test[:,0], configs['data']['sequence_length'])     
        # print(predictions_multiseq)
        # plot_results(predictions_pointbypoint, y_test)

    # 生成情景树
    tree_model = ScenarioTree(window=x_test[-1,:,:], model=BLmodel.model, T=2, num_sample=2)
    s_tree = tree_model.build_tree()
    
if __name__ == '__main__':
    main()