import scipy.io as sio
import numpy as np
import gurobipy as grb
from gurobipy import GRB
from core.Portfolio_model_solved_v1 import *

data = sio.loadmat(r'./data/reduced_cell_tree.mat')
reduced_cell_tree = data['C']

reduced_branch=[2,2]######################
reduced_node=np.cumprod(reduced_branch)
print(reduced_node[-1])
AX=1 # 情景树个数
d=7 #情景树节点维度
alpha=0.95
initial_wealth=300
stage=2
initial_meancvarobj=np.zeros((1,AX))
reduced_meancvarobj=np.zeros((1,AX))
initial_meancvardec=np.zeros((d-1,AX))
reduced_meancvardec=np.zeros((d-1,AX))
initial_mvobj = np.zeros((1, AX))
reduced_mvobj = np.zeros((1, AX))
initial_mvdec=np.zeros((d-1,AX))
reduced_mvdec=np.zeros((d-1,AX))
obj_diff=np.zeros((1,AX))
print(stage)
for i in range(AX):

    reduced_meancvar,reduced_x0_values = solve_portfolio_structure_tree_meancvar_decesion(d, reduced_cell_tree, initial_wealth,reduced_node, stage, reduced_branch,alpha)
    reduced_meancvarobj[0, i] = reduced_meancvar
    reduced_meancvardec[:, i] = reduced_x0_values
    print("reduced meancvarobj", reduced_meancvarobj)

