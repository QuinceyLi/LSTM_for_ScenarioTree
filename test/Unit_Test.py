# import sys
# sys.path.append('../..')
from core.data_processor import DataLoader
# import core
import json
import os

class DataLoader_test():
    def __init__(self, data):
        self.test = {}
        self.dataloader = data

    def test(self):
        print("-----测试DataLoader_test------")


    def get_test_data(self, seq_len, normalise):
        x,y = self.data.get_test_data(seq_len, normalise)
        print(x.shape)


configs = json.load(open('config_2.json', 'r'))
data = DataLoader(
    os.path.join('data', configs['data']['filename']),
    configs['data']['train_test_split'],
    configs['data']['columns'],
    configs['data']['output_idx']
)
data_test = DataLoader_test(data)
data_test.test()