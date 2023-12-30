# import sys
# sys.path.append('../..')
from core.data_processor import DataLoader
from core.tree import ScenarioTree
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
from treelib import Tree
from tqdm import tqdm
from numpy import newaxis
from core.model import Model, Bayesian_LSTM
from scipy.io import savemat
class test_ScenarioTree():
    def __init__(self, window: np.array, model, T: int, branch: list, n_stock: 5, model_name: str) -> None:
        """
        window: 初始的时间窗
        model: 已训练的模型
        T: 要生成T个时间点的情景树
        branch: 每个时间点的采样数,[1,3,4,5]
        n_stock: 股票种类
        model_name: 文件名
        """
        self.window = window
        self.model = model
        self.n_stock = n_stock
        self.T = T
        self.branch = branch
        self.model_name = model_name

    def generate_sample(self, mu, sigma, num_sample) -> np.array:
        """
        mu: [mu1, .. ,mu5]
        sigma: [s1, .. ,s5]
        return: [[y11, .. ,y15], .., [y_sample1, .. , y_sample5]]
        """
        samples = np.zeros((num_sample, self.n_stock))
        # print(samples.shape,"mu:",mu.shape)
        # print(np.random.normal(mu[0],sigma[0],self.num_sample))
        for i in range(self.n_stock):
            samples[:, i] = np.random.normal(mu[i], sigma[i], num_sample)

        return samples

    def process_savedata(self, save_data):
        """
        为了后续计算，需要把save_data做特殊的格式处理
        """
        new_data = []
        branch1 = np.cumprod(self.branch[1:])
        branch2 = np.cumsum(branch1)
        for idx, b in enumerate(branch2):
            if idx == 0:
                prob_col = [(1 / branch1[idx])] * branch1[idx]  # 概率列，每一个样本都是等概率生成的
                random_col = [0] * branch1[idx]  # 凑数
                # new_data.append(save_data[:b])
                new_data.append(np.insert(save_data[:b], save_data[:b].shape[1], [prob_col, random_col], axis=1))
            else:
                prob_col = [(1 / branch1[idx])] * branch1[idx]
                random_col = [0] * branch1[idx]
                new_data.append(np.insert(save_data[branch2[idx - 1]: b], save_data[branch2[idx - 1]: b].shape[1],
                                          [prob_col, random_col], axis=1))
        return new_data

    def get_branch(self):
        """
        计算branch的相关向量
        """
        branch0 = self.branch
        branch1 = np.cumprod(branch0)
        branch2 = np.cumsum(branch1)  # [1,3,5] -> [1,3,15] -> [1,4,19]
        return branch1, branch2

    def build_multi_trees(self, n_tree: int, skip: int, data: np.array) -> Tree:
        """
        构造多棵情景树
        data: data_test
        """
        initial_window = self.window
        window = np.vstack((initial_window, data))
        multi_tree = []
        print("\n\n-----------测试生成多个情景树:-------------\n\n")
        for i in tqdm(range(n_tree), desc='正在生成情景树:', position=0, leave=True):
            m_tree, save_data = self.build_tree(SAVE=False)
            # print("---")
            # print(m_tree)
            # np.array(m_tree)
            multi_tree.append(self.process_savedata(np.array(save_data)))
            # print("----")
            # multi_tree.append(self.process_savedata(np.array(self.build_tree(SAVE=False))))
            try:
                self.window = window[(i + 1) * skip:(i + 1) * skip + len(self.window), :]
            except Exception as e:
                print(e)

        filename = "multitree_" + self.model_name + ".mat"
        savemat(filename, {'tree': multi_tree})

        return multi_tree, "multitree_" + self.model_name + ".mat"

    def process_savedata(self, save_data):
        """
        为了后续计算，需要把save_data做特殊的格式处理
        """
        new_data = []
        branch1 = np.cumprod(self.branch[1:])
        branch2 = np.cumsum(branch1)
        for idx, b in enumerate(branch2):
            if idx == 0:
                prob_col = [(1 / branch1[idx])] * branch1[idx]  # 概率列，每一个样本都是等概率生成的
                random_col = [0] * branch1[idx]  # 凑数
                # new_data.append(save_data[:b])
                new_data.append(np.insert(save_data[:b], save_data[:b].shape[1], [prob_col, random_col], axis=1))
            else:
                prob_col = [(1 / branch1[idx])] * branch1[idx]
                random_col = [0] * branch1[idx]
                new_data.append(np.insert(save_data[branch2[idx - 1]: b], save_data[branch2[idx - 1]: b].shape[1],
                                          [prob_col, random_col], axis=1))
        return new_data

    def save_data2file(self, save_data, filename):
        """
        把情景树中的节点数据按顺序存入npy
        """
        # np.save(filename, save_data)

        processed_data = self.process_savedata(np.array(save_data))

        savemat(filename, {'tree': processed_data})

        print("Save data to file successfully!\n")

    def build_tree(self, SAVE=True) -> Tree:
        s_tree = Tree()
        s_tree.create_node(tag=0, identifier=0, data="ini_window")  # root

        nodes_tag = [0]  # 添加根节点tag

        save_data = []

        for i in range(self.T):
            branch1, branch2 = self.get_branch()
            pre_nodes_tags = nodes_tag[-branch1[i]:]

            for idx, node_tag in enumerate(pre_nodes_tags):
                curr_window = self.backtracking_window(node_tag, s_tree)
                out = self.model.predict(curr_window[newaxis, :, :], verbose=0)
                mu, sigma = tf.split(out, 2, axis=1)
                mu = tf.squeeze(mu, axis=1)
                sigma = tf.squeeze(sigma, axis=1)
                mu = mu.numpy()
                sigma = sigma.numpy()
                samples = self.generate_sample(mu[0], sigma[0], self.branch[i + 1])
                print(f"上一个节点的tag是{node_tag},它产生的sample是{samples}")
                for j in range(self.branch[i + 1]):
                    label = pre_nodes_tags[-1] + idx * self.branch[i + 1] + j + 1
                    s_tree.create_node(tag=label, identifier=label, parent=node_tag, data=samples[j])

                    nodes_tag.append(label)
                    save_data.append(samples[j])
        self.save_data2file(save_data, self.model_name + ".mat")

        # print("Build ScenarioTree successfully!\n")

        return s_tree, save_data

    def backtracking_window(self, node_tag: int, s_tree: Tree) -> np.array:
        """
        node_tag: 节点标签
        s_tree: 情景树
        回溯根节点到node_tag,以获取时间窗
        return: new_window
        """
        path2root = s_tree.rsearch(node_tag)
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

print("\n\n--------测试ScenarioTree----------")
train_dataset,x_train, y_train = data.get_train_data(
    seq_len=configs['data']['sequence_length'],
    batch_size=configs['training']['batch_size'],
    normalise=configs['data']['normalise']
)

BLmodel = Bayesian_LSTM()
mymodel = BLmodel.build_model(configs)

tsT = test_ScenarioTree(window=x_train[-1,:,:], model=mymodel, T=2, branch=[1,2,3],n_stock=5,model_name='test')
stree = tsT.build_tree()
tsT.build_multi_trees(2,2,data.data_test)

########################
from core.utils import plus_one,get_wealth
import pandas as pd
import matplotlib.pyplot as plt
from core.Portfolio_model_solved import *
import scipy.io as sio

def backtesting(mat_dict: dict, wealth_change_dict: dict, test_data: np.array, decision_method: str) -> dict:
    """

    """
    for model_name, mat_name in mat_dict.items():
        print("正在使用{}模型求解{}对应的优化问题...\n".format(decision_method, model_name))

        tree_data = sio.loadmat(mat_name)

        M_tree = tree_data['tree']
        # print(M_tree[0].shape)

        reduced_branch = [15, 8, 5]
        reduced_node = np.cumprod(reduced_branch)

        wealth = [300]
        wealth_change_list = []
        skip = 4
        print(len(M_tree))
        for i in range(len(M_tree)):
            print(f"正在求解第{i + 1}个情景树,共{len(M_tree)}个情景树", end='\r')
            M_tree[i] = plus_one(M_tree[i])
            reduced_cell_tree = np.array([M_tree[i]])
            # print(reduced_cell_tree)
            # 设置参数
            AX = 1  # 情景树个数
            d = 7  # 情景树节点维度
            alpha = 0.95
            initial_wealth = wealth[-1]
            stage = len(reduced_branch)
            initial_meancvarobj = np.zeros((1, AX))
            reduced_meancvarobj = np.zeros((1, AX))
            initial_meancvardec = np.zeros((d - 1, AX))
            reduced_meancvardec = np.zeros((d - 1, AX))
            initial_mvobj = np.zeros((1, AX))
            reduced_mvobj = np.zeros((1, AX))
            initial_mvdec = np.zeros((d - 1, AX))
            reduced_mvdec = np.zeros((d - 1, AX))
            obj_diff = np.zeros((1, AX))

            # Gurobi求解
            if decision_method == "meancvar":
                obj, x0, x = solve_portfolio_structure_tree_meancvar_decesion(d, reduced_cell_tree, initial_wealth,
                                                                              reduced_node, stage, reduced_branch,
                                                                              alpha)
            elif decision_method == "cvar":
                obj, x0, x = solve_portfolio_structure_tree_cvar_decesion(d, reduced_cell_tree, initial_wealth,
                                                                          reduced_node, stage, reduced_branch, alpha)
            elif decision_method == "quar":
                obj, x0, x = solve_portfolio_structure_tree_quar_decesion(d, reduced_cell_tree, initial_wealth,
                                                                          reduced_node, stage, reduced_branch, alpha)
            elif decision_method == "mv":
                obj, x0, x = solve_portfolio_structure_tree_mv(d, reduced_cell_tree, initial_wealth, reduced_node,
                                                               stage, reduced_branch, alpha)
            # print("obj", obj, "\nx0:", x0, "\nx:", x)
            print(f"x0:{x0}")

            wealth_change = get_wealth(x0, i * skip, test_data)
            # wealth.append(np.sum(next_wealth))
            wealth.append(wealth_change[-1])
            wealth_change_list.append(wealth_change)

        wealth_change_dict[model_name] = wealth_change_list

    return wealth_change_dict

# 读取情景树数据,做回测
print("\n\n-------正在测试回测代码----------\n\n")

mat_dict = {
            "ARMA-GARCH": "../data/ag_tree_04-Nov-2023 10_49_47.mat",
            "kmeans": "../data/kmeans_tree_04-Nov-2023 13_40_44.mat",
            "moment_matching": "../data/mm_tree_04-Nov-2023 16_49_08.mat",
            "NN": "../data/multitree_13112023-151552-e20.mat"
            }

wealth_change_dict = {
    "ARMA-GARCH": [],
    "kmeans": [],
    "moment_matching": [],
    "NN": []
}

configs = json.load(open('../config_2.json', 'r'))
mdata = DataLoader(
    os.path.join('../data', configs['data']['filename']),
    configs['data']['train_test_split'],
    configs['data']['columns'],
    configs['data']['output_idx']
)
decision_method = ["meancvar","mv"]
d = decision_method[0]
wealth_change_dict = backtesting(mat_dict, wealth_change_dict, mdata.data_test, d)
w = wealth_change_dict['NN']
w = np.array(w)
periods = w.shape[0] * w.shape[1]

wealth_df = pd.DataFrame(pd.date_range(start='2021-01-01', periods=periods, freq='D'), columns=['time'])
wealth_df.set_index('time', inplace=True)

for model_name, wealth_change_list in wealth_change_dict.items():
    wealth_change_list = np.hstack(wealth_change_list)
    wealth_df[model_name] = wealth_change_list

wealth_df.plot(figsize=(12, 8), title=d)
plt.show()
