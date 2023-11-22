# 情景树
import numpy as np
from core.model import Bayesian_LSTM
from treelib import Tree
from numpy import newaxis
import tensorflow as tf
import json
from scipy.io import savemat


class ScenarioTree():
    def __init__(self, window: np.array, model, T: int, num_sample: int, model_name: str) -> None:
        """
        window: 初始的时间窗
        model: 已训练的模型
        mu, sigma: 根据初始时间窗预测的下一时间点的分布的均值，标准差
        T: 要生成T个时间点的情景树
        num_sample: 每个时间点的采样数
        name: 文件名
        """
        self.window = window
        self.model = model
        self.T = T
        self.num_sample = num_sample
        self.model_name = model_name

    def generate_sample(self, mu, sigma) -> np.array:
        """
        mu: [mu1, .. ,mu5]
        sigma: [s1, .. ,s5]
        return: [[y11, .. ,y15], .., [y_sample1, .. , y_sample5]]
        """
        samples = np.zeros((self.num_sample, 5))
        # print(samples.shape,"mu:",mu.shape)
        # print(np.random.normal(mu[0],sigma[0],self.num_sample))
        for i in range(5):
            samples[:,i] = np.random.normal(mu[i],sigma[i],self.num_sample)

        return samples
    
    def process_savedata(self, save_data):
        """
        为了后续计算，需要把save_data做特殊的格式处理
        """
        new_data = []
        branch = [self.num_sample] * self.T # 情景树每个节点的分支数相同 [2,2,2]
        branch1 = np.cumprod(branch)
        branch2 = np.cumsum(branch1) # [2,2,2] -> [2,4,8] -> [2,6,14]
        for idx,b in enumerate(branch2):
            if idx == 0:
                prob_col = [(1/branch1[idx])] * branch1[idx] # 概率列，每一个样本都是等概率生成的
                random_col = [0] * branch1[idx] # 凑数
                # new_data.append(save_data[:b])
                new_data.append(np.insert(save_data[:b], save_data[:b].shape[1], [prob_col,random_col], axis=1))
            else:
                prob_col = [(1/branch1[idx])] * branch1[idx]
                random_col = [0] * branch1[idx]
                new_data.append(np.insert(save_data[branch2[idx-1]: b], save_data[branch2[idx-1]: b].shape[1],[prob_col,random_col], axis=1))
        return new_data
    
    def save_data2file(self, save_data, filename):
        """
        把情景树中的节点数据按顺序存入npy
        """
        # np.save(filename, save_data)

        processed_data = self.process_savedata(np.array(save_data))

        savemat(filename, {'tree': processed_data})

        print("Save data to file successfully!\n")

    def load_data(self, filename):
        """
        从npy中读取数据
        """
        save_data = np.load(filename)
        print("Load data from file successfully!")
        return save_data
    
    def tree_to_dict(self, tree, nid):
        """
        将情景树转换为dict,从而保存成json
        """
        node = tree[nid]
        if nid in [leaf.identifier for leaf in tree.leaves()]:
            return {node.tag: {}}
        else:
            tree_dict = {node.tag: {}}
            children = tree.children(nid)
            for child in children:
                tree_dict[node.tag].update(self.tree_to_dict(tree, child.identifier))
            return tree_dict
        
    def dict_to_tree(self,tree_dict, tree=None, parent=None):
        if tree is None:
            tree = Tree()
            parent = tree.root
        for k, v in tree_dict.items():
            if isinstance(v, dict):
                child = tree.create_node(k, parent=parent)
                self.dict_to_tree(v, tree=tree, parent=child.identifier)
            else:
                tree.create_node(k, parent=parent)
        return tree
    
    def build_tree(self) -> Tree:
        s_tree = Tree()
        s_tree.create_node(tag=0, identifier=0 ,data="ini_window") # root

        nodes_tag = [0] # 添加根节点tag

        save_data = []

        for i in range(self.T):
            pre_nodes_tags = nodes_tag[- self.num_sample ** i : ]

            # 对每一个pre_node,求出其num_samples个分支节点
            for idx, node_tag in enumerate(pre_nodes_tags):
                # 回溯pre_node节点到根节点，获取时间窗
                curr_window = self.backtracking_window(node_tag, s_tree)

                out = self.model.predict(curr_window[newaxis,:,:]) 
                mu,sigma = tf.split(out, 2, axis=1)
                mu = tf.squeeze(mu, axis=1)
                sigma = tf.squeeze(sigma, axis=1)
                # 将mu从tensor转换为ndarray
                mu = mu.numpy()
                sigma = sigma.numpy()
                # print("mu.shape:",mu.shape, "sigma.shape:",sigma.shape)

                samples = self.generate_sample(mu[0], sigma[0])# mu: (5,)
                for j in range(self.num_sample):
                    s_tree.create_node(tag=pre_nodes_tags[-1] + idx*self.num_sample + j + 1, identifier=pre_nodes_tags[-1] + idx*self.num_sample + j + 1, parent=node_tag, data=samples[j])

                    nodes_tag.append(pre_nodes_tags[-1] + idx*self.num_sample + j + 1)
                    save_data.append(samples[j])

        print("Build ScenarioTree successfully!\n")

        self.save_data2file(save_data, "./data/" + self.model_name + ".mat")
        tree_dict = self.tree_to_dict(s_tree, s_tree.root)
        with open("./data/" + self.model_name + ".json","w") as f:
            json.dump(tree_dict, f)
        print("Save Tree to json sucessfully!\n")

        return s_tree
    
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

        if len(temp) < 2: # 只有根节点，则直接返回window
            return self.window
        
        # p的顺序为从子节点到根节点，所以反转列表temp
        temp = temp[::-1]
        # print(temp)
        
        temp = temp[1:] # 去掉根节点
        new_window = np.vstack([self.window, temp])
        return new_window[len(temp):]

        

    