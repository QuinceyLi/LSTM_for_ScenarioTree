import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from core.Portfolio_model_solved import *

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))

# 绘图展示结果 
    
def plot_results_multiple(predicted_data, true_data, prediction_len):     
    fig = plt.figure(facecolor='white')     
    ax = fig.add_subplot(111)     
    ax.plot(true_data, label='True Data')     
    plt.legend()     
    for i, data in enumerate(predicted_data):         
        padding = [None for p in range((i+1) * prediction_len)]         
        plt.plot(padding + data, label='Prediction')     
        plt.legend()     
        #plt.show()     
        plt.savefig('./pic/results_multiple_2.png')

#####回测######
		
def get_wealth(porforlio: np.array, time_idx: int, return_data: np.array):
    """
    porforlio: 根节点的投资组合
    time_idx: 当前时间点
    return_data: 股票收益率数据

    return: close_data: 当前时间点的投资在skip时间段后的资产
    """
    skip = 4 # 根节点的投资组合持仓时间
    norisk = porforlio[0] # 无风险资产
    stock = porforlio[1:]
    interest_rate = 0.001 # 无风险利率
    wealth_change = np.zeros(skip) # 记录总的资产的变化
    return_data = return_data[time_idx:time_idx + skip,:] + 1

    for d in range(skip):
        # w1 = norisk * ( (1+interest_rate) ** skip )
        w1 = norisk *  ( (1+interest_rate) ** (d+1) )
        w2 = np.dot(stock, return_data[d,:])
        stock = stock * return_data[d,:]
        wealth_change[d] = w1 + w2

    return wealth_change

def plus_one(tree_arr: np.array) -> np.array:
    """
    为所有情景树节点加1,概率列不加1
    """
    for sub_tree in tree_arr:
        sub_tree[:,:-2] = sub_tree[:,:-2] + 1

    return tree_arr
        
def backtesting(mat_dict:dict, wealth_change_dict:dict, test_data:np.array, decision_method:str) -> dict:
    """
    
    """
    for model_name, mat_name in mat_dict.items():
        print("正在使用{}模型求解{}对应的优化问题...\n".format(decision_method,model_name))

        tree_data = sio.loadmat('./data/' + mat_name)

        M_tree = tree_data['tree']
        # print(M_tree[0].shape)

        reduced_branch= [15,8,5]
        reduced_node=np.cumprod(reduced_branch)

        wealth = [300]
        wealth_change_list = []
        skip = 4

        for i in range(len(M_tree)):
            print(f"正在求解第{i+1}个情景树,共{len(M_tree)}个情景树", end='\r')
            M_tree[i] = plus_one(M_tree[i])
            reduced_cell_tree=np.array([M_tree[i]])
            # print(reduced_cell_tree)
            # 设置参数
            AX=1 # 情景树个数
            d=7 #情景树节点维度
            alpha=0.95
            initial_wealth=wealth[-1]
            stage=len(reduced_branch)
            initial_meancvarobj=np.zeros((1,AX))
            reduced_meancvarobj=np.zeros((1,AX))
            initial_meancvardec=np.zeros((d-1,AX))
            reduced_meancvardec=np.zeros((d-1,AX))
            initial_mvobj = np.zeros((1, AX))
            reduced_mvobj = np.zeros((1, AX))
            initial_mvdec=np.zeros((d-1,AX))
            reduced_mvdec=np.zeros((d-1,AX))
            obj_diff=np.zeros((1,AX))

            # Gurobi求解
            if decision_method == "meancvar":
                obj,x0, x = solve_portfolio_structure_tree_meancvar_decesion(d, reduced_cell_tree, initial_wealth,reduced_node, stage, reduced_branch,alpha)
            elif decision_method == "cvar":
                obj,x0, x = solve_portfolio_structure_tree_cvar_decesion(d, reduced_cell_tree, initial_wealth,reduced_node, stage, reduced_branch,alpha)
            elif decision_method == "quar":
                obj,x0, x = solve_portfolio_structure_tree_quar_decesion(d, reduced_cell_tree, initial_wealth,reduced_node, stage, reduced_branch,alpha)
            elif decision_method == "mv":
                obj, x0, x = solve_portfolio_structure_tree_mv(d,reduced_cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha)
            # print("obj", obj, "\nx0:", x0, "\nx:", x)
            # print(f"x0:{x0}")     
            
            wealth_change = get_wealth(x0, i*skip, test_data)
            # wealth.append(np.sum(next_wealth))
            wealth.append(wealth_change[-1])
            wealth_change_list.append(wealth_change)

        wealth_change_dict[model_name] = wealth_change_list

    return wealth_change_dict

