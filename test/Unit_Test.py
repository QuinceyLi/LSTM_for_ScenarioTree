# import sys
# sys.path.append('../..')
from core.data_processor import DataLoader
# import core
import json
import os
import numpy as np

class DataLoader_test():
    def __init__(self, data):
        self.test_dict = {}
        self.dataloader = data

    def test(self):
        print("\n\n-----测试DataLoader------")
        print("\n测试get_test_data函数:")
        self.test_dict['get_test_data'] = self.test_get_test_data(5,False)
        print("\n测试get_train_data函数:")
        self.test_dict['get_train_data'] = self.test_get_train_data(5, 2, False)
        print("\n测试normalise_windows函数:")
        window_data = np.array([[[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]])
        self.test_dict['normalise_windows'] = self.test_normalise_windows(window_data)
        test_res = [r for r in self.test_dict.values()]
        if False in test_res:
            print("\n------DataLoader测试失败，存在未通过测试的单元------")
        else:
            print("\n------DataLoader测试成功，全部单元已通过测试------")

    def test_get_train_data(self, seq_len, batch_size, normalise):
        train_dataset, x, y = self.dataloader.get_train_data(seq_len, batch_size, normalise)
        # print(train_dataset,'\nx:\n', x,'y:\n', y)
        # print(np.vstack([x[0],y]))
        print("测试失败")
        print("失败原因：for i in range(self.len_test - seq_len) 最后一行数据丢失")
        return False

    def test_normalise_windows(self,window_data, single_window=False ):
        normalized_window = self.dataloader.normalise_windows(window_data, single_window)
        print("对同一股票的所有天数做标准化，现在感觉有点奇怪...")
        return False


    def test_get_test_data(self, seq_len, normalise):
        x,y = self.dataloader.get_test_data(seq_len, normalise)
        # print(x,'\n\n',y)
        new_data = []
        new_data.append(np.vstack([x[0],y]))
        new_data = np.array(new_data).flatten().reshape(-1, 5)
        # print("newdata:",new_data)
        # print("olddata:",self.dataloader.data_test)
        try:
            if np.equal(new_data, self.dataloader.data_test):
                print("测试通过")
                return True
        except Exception as e:
            pass
        print("测试失败")
        print("失败原因：for i in range(self.len_test - seq_len) 最后一行数据丢失")
        return False


configs = json.load(open('test_config.json', 'r'))
data = DataLoader(
    os.path.join('../data', configs['data']['filename']),
    configs['data']['train_test_split'],
    configs['data']['columns'],
    configs['data']['output_idx']
)
# print(data.data_test)
data_test = DataLoader_test(data)
data_test.test()

#############
import tensorflow as tf
print("\n\n---------测试bayesian_train:---------")
batch = 2
train_x_shape0 = batch
train_y_shape1 = 5
num_sample = 3
out = tf.constant([ [[1,2,3,4,5],[2,3,4,5,6]] , [[3,4,5,6,7], [1,2,3,4,5]]], dtype=tf.float32)
def test_bayesian_train(batch, train_x_shape_0, train_y_shape1, num_sample, out):
    print("网络的输出是:",out)
    mu,sigma = tf.split(out, 2, axis=1)
    print("\nsplit后,mu:",mu)
    mu = tf.squeeze(mu, axis=1)
    print("\nsqueeze后,mu:",mu)
    sigma = tf.squeeze(sigma, axis=1)
    sample_total = tf.zeros((num_sample, train_x_shape0, train_y_shape1))
    for t in range(num_sample):
        epsilon = tf.random.normal(tf.shape(sigma))
        sample = mu + tf.multiply(sigma, epsilon)
        sample_total = tf.tensor_scatter_nd_update(sample_total, [[t]], [sample])
        print("对batch的数据做采样,得到:",sample_total)
    sample_ave = tf.reduce_mean(sample_total, axis=0)
    print("\n\n对采样的数据取平均:",sample_ave)
    print("可以直接将预测的期望值计算Loss")
    return sample_ave,mu,sigma

sample_ave,mu,sigma = test_bayesian_train(batch, train_x_shape0, train_y_shape1, num_sample, out)

print("\n\n---------测试loss_fun---------")
def test_loss_fun(sample_ave,sigma):
    train_y = tf.constant([[5,6,7,8,9], [6,7,8,9,6]], dtype=tf.float32)
    print("\n标签y:", train_y)
    rows, cols = train_y.shape
    sample_y = sample_ave
    square_loss = tf.math.square((train_y - sample_y) / sigma)
    log_loss = 2 * tf.math.log(sigma)
    sample_loss = square_loss + log_loss
    print("\nsample_loss = square_loss + log_loss:",sample_loss)
    loss_val = tf.reduce_sum(sample_loss, axis=1) / (2 * cols)
    print("\n对5维的数据算均值loss_val:",loss_val)
    print("\n对Batch中每个数据取平均:",tf.reduce_mean(loss_val))
    print("测试成功")
test_loss_fun(sample_ave,sigma)

##################

class test_ScenarioTree():
    def __init__(self, stree) -> None:
        self.stree = stree

    def test_backtracking_window(self, node_tag):
        s_tree = self.stree
        path2root = s_tree.rsearch(node_tag)
        print(path2root)
        temp = []
        for p in path2root:
            temp.append(s_tree.get_node(p).data)

        if len(temp) < 2:  # 只有根节点，则直接返回window
            return self.window

        # p的顺序为从子节点到根节点，所以反转列表temp
        temp = temp[::-1]
        # print(temp)

        temp = temp[1:]  # 去掉根节点
        new_window = np.vstack([self.window, temp])
        return new_window[len(temp):]

    def test_build_tree(self):


