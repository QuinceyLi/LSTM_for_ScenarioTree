import scipy.io as sio
import numpy as np
import gurobipy as grb
from gurobipy import GRB
import Portfolio_model_solved
scenario_data=sio.loadmat(r'utility_kmedios_10times_201010_852_tree1_same_initial_tree.mat')
initial_tree_J=scenario_data['initial_cell_tree_all']
reduced_tree_J=scenario_data['reduced_cell_tree_all']
n_branch=scenario_data['n_branch']
n_branch0=n_branch[0,1:]
initial_node=np.cumprod(n_branch0)
print(n_branch0)
reduced_branch=[8,5,2]######################
reduced_node=np.cumprod(reduced_branch)
print(reduced_node[-1])
AX=10
d=9 #情景树节点维度
alpha=0.95
initial_wealth=300
stage=n_branch0.shape[0]

initial_cvarobj=np.zeros((1,AX))
reduced_cvarobj=np.zeros((1,AX))
initial_cvardec=np.zeros((d-1,AX))
reduced_cvardec=np.zeros((d-1,AX))
initial_meancvarobj=np.zeros((1,AX))
reduced_meancvarobj=np.zeros((1,AX))
initial_meancvardec=np.zeros((d-1,AX))
reduced_meancvardec=np.zeros((d-1,AX))
initial_quarobj=np.zeros((1,AX))
reduced_quarobj=np.zeros((1,AX))
initial_quardec=np.zeros((d-1,AX))
reduced_quardec=np.zeros((d-1,AX))
initial_mvobj = np.zeros((1, AX))
reduced_mvobj = np.zeros((1, AX))
initial_mvdec=np.zeros((d-1,AX))
reduced_mvdec=np.zeros((d-1,AX))
obj_diff=np.zeros((1,AX))
print(stage)
for i in range(AX):
    print(i)
    initial_cell_tree=initial_tree_J[0,i]
    reduced_cell_tree=reduced_tree_J[0,i]
    print(reduced_cell_tree[0,2][0,:])
    reduced_meancvar,reduced_x0_values = Portfolio_model_solved.solve_portfolio_structure_tree_meancvar_decesion(d, reduced_cell_tree, initial_wealth,reduced_node, stage, reduced_branch,alpha)
    reduced_meancvarobj[0, i] = reduced_meancvar
    reduced_meancvardec[:, i] = reduced_x0_values
    print("reduced meancvarobj", reduced_meancvarobj)


# sio.savemat('Python_utility_kmedios_10times_201010_852_cell_tree1_decesion_same_initial_tree',  {'reduced_meancvarobj': reduced_meancvarobj,'initial_meancvarobj': initial_meancvarobj,'reduced_meancvardec':reduced_meancvardec,'initial_meancvardec':initial_meancvardec,'meancvarobj_diff': meancvarobj_diff,'meancvardec_diff':meancvardec_diff,'reduced_mvobj': reduced_mvobj,'initial_mvobj': initial_mvobj,'reduced_mvdec':reduced_mvdec,'initial_mvdec':initial_mvdec,'mvobj_diff': mvobj_diff,'mvdec_diff':mvdec_diff})
#
#
#
