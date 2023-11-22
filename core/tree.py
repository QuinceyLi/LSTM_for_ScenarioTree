# 情景树
import numpy as np
from core.model import Bayesian_LSTM
from treelib import Tree
from numpy import newaxis
import tensorflow as tf
import json
from tqdm import tqdm
from scipy.io import savemat


class ScenarioTree():
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
            samples[:,i] = np.random.normal(mu[i],sigma[i], num_sample)

        return samples
    
    def process_savedata(self, save_data):
        """
        为了后续计算，需要把save_data做特殊的格式处理
        """
        new_data = []
        branch1 = np.cumprod(self.branch[1:])
        branch2 = np.cumsum(branch1)
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
    
    def get_branch(self):
        """
        计算branch的相关向量
        """
        branch0 =self.branch 
        branch1 = np.cumprod(branch0)
        branch2 = np.cumsum(branch1) # [1,3,5] -> [1,3,15] -> [1,4,19]
        return branch1, branch2

    def build_multi_trees(self, n_tree: int, skip: int, data:np.array) -> Tree:
        """
        构造多棵情景树
        data: data_test
        """
        initial_window = self.window 
        window = np.vstack((initial_window, data))
        multi_tree = []
        for i in tqdm(range(n_tree), desc='正在生成情景树:',position=0, leave=True):
            m_tree, save_data = self.build_tree(SAVE=False)
            # print("---")
            # print(m_tree)
            # np.array(m_tree)
            multi_tree.append(self.process_savedata(np.array(save_data)))
            # print("----")
            # multi_tree.append(self.process_savedata(np.array(self.build_tree(SAVE=False))))
            try:
                self.window = window[(i+1)*skip:(i+1)*skip + len(self.window),:]
            except Exception as e:
                print(e)
        filename = "./data/" + "multitree_" + self.model_name + ".mat"
        savemat(filename, {'tree': multi_tree})
        
        return multi_tree,"multitree_" + self.model_name + ".mat"


    def build_tree(self, SAVE=True) -> Tree:
        s_tree = Tree()
        s_tree.create_node(tag=0, identifier=0 ,data="ini_window") # root

        nodes_tag = [0] # 添加根节点tag

        save_data = []

        for i in range(self.T):
            branch1, branch2 = self.get_branch()
            pre_nodes_tags = nodes_tag[-branch1[i]:]

            for idx, node_tag in enumerate(pre_nodes_tags):
                curr_window = self.backtracking_window(node_tag, s_tree)
                out = self.model.predict(curr_window[newaxis,:,:], verbose=0)
                mu,sigma = tf.split(out,2,axis=1)
                mu = tf.squeeze(mu, axis=1)
                sigma = tf.squeeze(sigma, axis=1)
                mu = mu.numpy()
                sigma = sigma.numpy()
                samples = self.generate_sample(mu[0],sigma[0],self.branch[i+1])
                for j in range(self.branch[i+1]):
                    label = pre_nodes_tags[-1] + idx*self.branch[i+1] + j + 1
                    s_tree.create_node(tag=label, identifier=label, parent=node_tag, data=samples[j])

                    nodes_tag.append(label)
                    save_data.append(samples[j])

        # print("Build ScenarioTree successfully!\n")

        if SAVE:
            self.save_data2file(save_data, "./data/" + self.model_name + ".mat")
            tree_dict = self.tree_to_dict(s_tree, s_tree.root)
            with open("./data/" + self.model_name + ".json","w") as f:
                json.dump(tree_dict, f)
            print("Save Tree to json sucessfully!\n")

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

        if len(temp) < 2: # 只有根节点，则直接返回window
            return self.window
        
        # p的顺序为从子节点到根节点，所以反转列表temp
        temp = temp[::-1]
        # print(temp)
        
        temp = temp[1:] # 去掉根节点
        new_window = np.vstack([self.window, temp])
        return new_window[len(temp):]

        

    