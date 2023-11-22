 # 测试 
from core.model import Bayesian_LSTM
from core.tree import ScenarioTree
import json
from core.data_processor import DataLoader
import os

BNN = Bayesian_LSTM() 
BNN.load_model("./saved_models/24092023-094817-e1.h5")  

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

x_test, y_test = data.get_test_data(             seq_len=configs['data']['sequence_length'],             normalise=configs['data']['normalise']         ) 

s_tree = ScenarioTree(window=x_test[-1], model=BNN.model, T=2,num_sample=2)
new_tree = s_tree.build_tree()

old_data = s_tree.load_data("./data/Scenario_Tree_Data.npy")

old_tree = json.load(open("./data/tree.json","r"))

print(s_tree.dict_to_tree(old_tree))

print(old_data)
