def solve_portfolio_fan_tree_cvar(d,fan_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    leaf_node=reduced_node[-1]
    leaf_node=int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(stage,1,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(stage,leaf_node,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(leaf_node,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(leaf_node):
        for j in range(d-2):
            model.addConstr(x[0,j,i]==x0[0,j]*fan_tree[i,j]+b[0,j,i]-s[0,j,i])
            model.addConstr(s[0,j,i]<=x0[0,j]*fan_tree[i,j])
        model.addConstr(y[0,0,i]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(0,'*',i)+(1-c_s)*s.sum(0,'*',i))
        model.addConstr(z[0,i]==grb.quicksum(x0[0,k]*fan_tree[i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for j in range(leaf_node):
            for i in range(d-2):
                model.addConstr(x[t,i,j]==x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)]+b[t,i,j]-s[t,i,j])
                model.addConstr(s[t,i,j]<=x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)])
            model.addConstr(y[t,0,j]==y[t-1,0,j]*(1+interest_rate)-(1+c_b)*b.sum(t,'*',j)+(1-c_s)*s.sum(t,'*',j))
            model.addConstr(z[t,j]==grb.quicksum(x[t-1,k,j]*fan_tree[j,(d-2)*(t-1)+k] for k in range(d-2))+y[t-1,0,j]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[stage-1,i]<=zc+f[i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[k,0]*fan_tree[k,-3] for k in range(leaf_node)))

    model.setObjective(cvar, sense=grb.GRB.MINIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
######################################################################################################################
def solve_portfolio_fan_tree_meancvar(d,fan_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    leaf_node=reduced_node[-1]
    leaf_node=int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(stage,1,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(stage,leaf_node,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(leaf_node,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(leaf_node):
        for j in range(d-2):
            model.addConstr(x[0,j,i]==x0[0,j]*fan_tree[i,j]+b[0,j,i]-s[0,j,i])
            model.addConstr(s[0,j,i]<=x0[0,j]*fan_tree[i,j])
        model.addConstr(y[0,0,i]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(0,'*',i)+(1-c_s)*s.sum(0,'*',i))
        model.addConstr(z[0,i]==grb.quicksum(x0[0,k]*fan_tree[i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for j in range(leaf_node):
            for i in range(d-2):
                model.addConstr(x[t,i,j]==x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)]+b[t,i,j]-s[t,i,j])
                model.addConstr(s[t,i,j]<=x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)])
            model.addConstr(y[t,0,j]==y[t-1,0,j]*(1+interest_rate)-(1+c_b)*b.sum(t,'*',j)+(1-c_s)*s.sum(t,'*',j))
            model.addConstr(z[t,j]==grb.quicksum(x[t-1,k,j]*fan_tree[j,(d-2)*(t-1)+k] for k in range(d-2))+y[t-1,0,j]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[stage-1,i]<=zc+f[i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[k,0]*fan_tree[k,-3] for k in range(leaf_node)))

    model.setObjective(grb.quicksum(z[stage-1,k]*fan_tree[k,-3] for k in range(leaf_node))+cvar, sense=grb.GRB.MINIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
################################################################################################
def solve_portfolio_fan_tree_quar(d,fan_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    leaf_node=reduced_node[-1]
    leaf_node=int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(stage,1,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(stage,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(stage,leaf_node,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(leaf_node):
        for j in range(d-2):
            model.addConstr(x[0,j,i]==x0[0,j]*fan_tree[i,j]+b[0,j,i]-s[0,j,i])
            model.addConstr(s[0,j,i]<=x0[0,j]*fan_tree[i,j])
        model.addConstr(y[0,0,i]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(0,'*',i)+(1-c_s)*s.sum(0,'*',i))
        model.addConstr(z[0,i]==grb.quicksum(x0[0,k]*fan_tree[i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for j in range(leaf_node):
            for i in range(d-2):
                model.addConstr(x[t,i,j]==x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)]+b[t,i,j]-s[t,i,j])
                model.addConstr(s[t,i,j]<=x[t-1,i,j]*fan_tree[j,i+(t-1)*(d-2)])
            model.addConstr(y[t,0,j]==y[t-1,0,j]*(1+interest_rate)-(1+c_b)*b.sum(t,'*',j)+(1-c_s)*s.sum(t,'*',j))
            model.addConstr(z[t,j]==grb.quicksum(x[t-1,k,j]*fan_tree[j,(d-2)*(t-1)+k] for k in range(d-2))+y[t-1,0,j]*(1+interest_rate))
    gamma1 = 1
    gamma2 = 700
    model.setObjective(-grb.quicksum(fan_tree[i,-3]*(gamma2*z[stage-1,i]-gamma1*(z[stage-1,i]*z[stage-1,i])) for i in range(leaf_node)))
    model.optimize()
    obj = model.ObjVal
    return obj
######################################################################################################################
######################################################################################################################
################################################以上均是求解扇形树代码####################################################
######################################################################################################################
######################################################################################################################
def solve_portfolio_structure_tree_cvar(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(cvar, sense=grb.GRB.MINIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
def solve_portfolio_structure_tree_cvar_decesion(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(cvar, sense=grb.GRB.MINIMIZE)
    model.optimize()
    x0_values=np.zeros(d-1)
    x0_values[0]=y0[0,0].x
    for i in range(d-2):
        x0_values[1+i]=x0[0,i].x
    obj = model.ObjVal
    return obj,x0_values
def solve_portfolio_structure_tree_meancvar(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(grb.quicksum(z[int(np.sum(reduced_node[0:stage-1]))+k,0] * cell_tree[0, -1][k,-2] for k in range(leaf_node))-cvar, sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
def solve_portfolio_structure_tree_meancvar_decesion(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(grb.quicksum(z[int(np.sum(reduced_node[0:stage-1]))+k,0] * cell_tree[0, -1][k,-2] for k in range(leaf_node))-cvar, sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    x0_values = np.zeros(d - 1)
    x0_values[0] = y0[0, 0].x
    for i in range(d - 2):
        x0_values[1 + i] = x0[0, i].x
    obj = model.ObjVal
    return obj, x0_values
####################################################################################################################
def solve_portfolio_structure_tree_meancvar_shortselling(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    f=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    zc=model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar=model.addVar(vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        # model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            # model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    # model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    # model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))

    for i in range(leaf_node):
        model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(grb.quicksum(z[int(np.sum(reduced_node[0:stage-1]))+k,0] * cell_tree[0, -1][k,-2] for k in range(leaf_node))-cvar, sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
####################################################################################################################
def solve_portfolio_structure_tree_quar(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(gamma2*z[np.sum(reduced_node[0:stage-1])+i, 0]-gamma1*(z[np.sum(reduced_node[0:stage-1])+i,0]*z[np.sum(reduced_node[0:stage-1])+i,0]) ) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
def solve_portfolio_structure_tree_quar_decesion(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(gamma2*z[np.sum(reduced_node[0:stage-1])+i, 0]-gamma1*(z[np.sum(reduced_node[0:stage-1])+i,0]*z[np.sum(reduced_node[0:stage-1])+i,0]) ) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    x0_values = np.zeros(d - 1)
    x0_values[0] = y0[0, 0].x
    for i in range(d - 2):
        x0_values[1 + i] = x0[0, i].x
    obj = model.ObjVal
    return obj, x0_values
####################################################################################################################
def solve_portfolio_structure_tree_quar_shortselling(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=grb.GRB.INFINITY)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        # model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            # model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    # model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    # model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(gamma2*z[np.sum(reduced_node[0:stage-1])+i, 0]-gamma1*(z[np.sum(reduced_node[0:stage-1])+i,0]*z[np.sum(reduced_node[0:stage-1])+i,0]) ) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
####################################################################################################################
def solve_portfolio_structure_tree_mv(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    me_v = model.addVar(vtype=grb.GRB.CONTINUOUS)
    model.addConstr(me_v==grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))
    model.setObjective(me_v-grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)

    # model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node))- grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]) for i in range(leaf_node)))*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))) for i in range(leaf_node)), sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    # z_so=np.zeros((N_t,1))
    # for i in range(N_t):
    #     z_so[i,0]=z[i,0].x
    # me_vso=me_v.x
    return obj#, z_so, me_vso
def solve_portfolio_structure_tree_mv_decesion(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    me_v = model.addVar(vtype=grb.GRB.CONTINUOUS)
    model.addConstr(me_v==grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))
    model.setObjective(me_v-grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)

    # model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node))- grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]) for i in range(leaf_node)))*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))) for i in range(leaf_node)), sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    x0_values = np.zeros(d - 1)
    x0_values[0] = y0[0, 0].x
    for i in range(d - 2):
        x0_values[1 + i] = x0[0, i].x
    obj = model.ObjVal
    return obj, x0_values
###########################################################################################################
def solve_portfolio_structure_tree_mv_shortselling(d,cell_tree,initial_wealth,reduced_node,stage,reduced_branch,alpha):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t = np.sum(reduced_node)
    N_t=int(N_t)
    leaf_node = reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:reduced_node[0],0]=initial_wealth*(1+g_rate)*np.ones(reduced_node[0])
    for t in range(stage-1):
        g[np.sum(reduced_node[0:t+1]):np.sum(reduced_node[0:t+2]),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(reduced_node[t+1])
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb= -grb.GRB.INFINITY)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        # model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(reduced_node[0]):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            # model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(reduced_node[t]):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[reduced_node[0]+i,j]==x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[reduced_node[0]+i,j]-s[reduced_node[0]+i,j])
                    # model.addConstr(s[reduced_node[0]+i,j]<=x[math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[np.sum(reduced_node[0:t])+i,j]==x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j]+b[np.sum(reduced_node[0:t])+i,j]-s[np.sum(reduced_node[0:t])+i,j])
                    # model.addConstr(s[np.sum(reduced_node[0:t])+i,j]<=x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[reduced_node[0]+i,0]==y[math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(reduced_node[0]+i,'*')-(1+c_b)*b.sum(reduced_node[0]+i,'*'))
                model.addConstr(z[reduced_node[0]+i,0]==grb.quicksum(x[math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
            else:
                model.addConstr(y[np.sum(reduced_node[0:t])+i,0]==y[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t])+i,'*')-(1+c_b)*b.sum(np.sum(reduced_node[0:t])+i,'*'))
                model.addConstr(z[np.sum(reduced_node[0:t])+i,0]==grb.quicksum(x[np.sum(reduced_node[0:t-1])+math.floor(i/reduced_branch[t]),k]*cell_tree[0,t][i,k] for k in range(d-2))+y[np.sum(reduced_node[0:t-1])+np.floor(i/reduced_branch[t]),0]*(1+interest_rate))
    gamma1=1
    gamma2=700
    me_v = model.addVar(vtype=grb.GRB.CONTINUOUS)
    model.addConstr(me_v==grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))
    model.setObjective(me_v-grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)*(z[np.sum(reduced_node[0:stage-1])+i,0]-me_v)) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)

    # model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node))- grb.quicksum((cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i,0]) for i in range(leaf_node)))*(z[np.sum(reduced_node[0:stage-1])+i,0]-grb.quicksum(cell_tree[0,-1][i,-2]*(z[np.sum(reduced_node[0:stage-1])+i, 0]) for i in range(leaf_node)))) for i in range(leaf_node)), sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    xv=np.zeros((N_t,d-2))
    for i in range(N_t):
        for j in range(d-2):
            xv[i,j]=x[i,j].x
    return obj, xv
################################ SBRM CELL TREE QUAR###################################################
def solve_portfolio_structure_tree_quarSBRM(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))

    # for i in range(leaf_node):
    #     model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    # model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-3] for k in range(leaf_node)))
    gamma1=1
    gamma2=700
    model.setObjective(grb.quicksum(cell_tree[0,-1][i,-2]*(gamma2*z[int(np.sum(reduced_node[0:stage-1]))+i, 0]-gamma1*(z[int(np.sum(reduced_node[0:stage-1]))+i,0]*z[int(np.sum(reduced_node[0:stage-1]))+i,0]) ) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
####################################################################################################################

###########################################################################################################
################################ SBRM CELL TREE CVAR###################################################
def solve_portfolio_structure_tree_CVARSBRM(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS)
    f = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS, lb=0)
    zc = model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar = model.addVar(vtype=grb.GRB.CONTINUOUS)
    ######### cvar variables#############

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))
    for i in range(leaf_node):
        model.addConstr(g[int(np.sum(reduced_node[0:stage-1]))+i,0]-z[int(np.sum(reduced_node[0:stage-1]))+i,0]<=zc+f[int(np.sum(reduced_node[0:stage-1]))+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[int(np.sum(reduced_node[0:stage-1]))+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))

    model.setObjective(cvar, sense=grb.GRB.MINIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj

###########################################################################################################
################################ SBRM CELL TREE CVAR###################################################
def solve_portfolio_structure_tree_MEANCVARSBRM(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS)
    f = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS, lb=0)
    zc = model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar = model.addVar(vtype=grb.GRB.CONTINUOUS)
    ######### cvar variables#############

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))
    for i in range(leaf_node):
        model.addConstr(g[int(np.sum(reduced_node[0:stage-1]))+i,0]-z[int(np.sum(reduced_node[0:stage-1]))+i,0]<=zc+f[int(np.sum(reduced_node[0:stage-1]))+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[int(np.sum(reduced_node[0:stage-1]))+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))
    model.setObjective(grb.quicksum(z[int(np.sum(reduced_node[0:stage-1]))+k,0] * cell_tree[0, -1][k,-2] for k in range(leaf_node))-cvar,
                       sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
def solve_portfolio_structure_tree_MEANCVARSBRM_decesion(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS)
    f = model.addVars(N_t, 1, vtype=grb.GRB.CONTINUOUS, lb=0)
    zc = model.addVar(vtype=grb.GRB.CONTINUOUS)
    cvar = model.addVar(vtype=grb.GRB.CONTINUOUS)
    ######### cvar variables#############

    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))
    for i in range(leaf_node):
        model.addConstr(g[int(np.sum(reduced_node[0:stage-1]))+i,0]-z[int(np.sum(reduced_node[0:stage-1]))+i,0]<=zc+f[int(np.sum(reduced_node[0:stage-1]))+i,0])

    model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[int(np.sum(reduced_node[0:stage-1]))+k,0]*cell_tree[0,-1][k,-2] for k in range(leaf_node)))
    model.setObjective(grb.quicksum(z[int(np.sum(reduced_node[0:stage-1]))+k,0] * cell_tree[0, -1][k,-2] for k in range(leaf_node))-cvar,
                       sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    x0_values = np.zeros(d - 1)
    x0_values[0] = y0[0, 0].x
    for i in range(d - 2):
        x0_values[1 + i] = x0[0, i].x
    obj = model.ObjVal
    return obj, x0_values

def solve_portfolio_structure_tree_mvSBRM(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))

    # for i in range(leaf_node):
    #     model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    # model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-3] for k in range(leaf_node)))
    gamma1=1
    gamma2=700
    me_v = model.addVar(vtype=grb.GRB.CONTINUOUS)
    model.addConstr(me_v ==grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)))
    model.setObjective(me_v-grb.quicksum((cell_tree[0,-1][i,-2]*((z[int(np.sum(reduced_node[0:stage-1]))+i, 0]-me_v)*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0]-me_v))) for i in range(leaf_node)), sense=grb.GRB.MAXIMIZE)

    # model.setObjective(grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)) - grb.quicksum((cell_tree[0,-1][i,-2]*((z[int(np.sum(reduced_node[0:stage - 1]))+i,0]-grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)))*(z[int(np.sum(reduced_node[0:stage - 1])) + i, 0] - grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i,0])) for i in range(leaf_node))))) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    return obj
def solve_portfolio_structure_tree_mvSBRM_decesion(d,cell_tree, initial_wealth, n_br, stage, n_br_father, alpha,N_t,reduced_node):
    import gurobipy as grb
    import numpy as np
    import math
    c_s = 0.001
    c_b = 0.001
    interest_rate = 0.001
    N_t=int(N_t)
    leaf_node=reduced_node[-1]
    leaf_node = int(leaf_node)
    x0_ = initial_wealth / (d - 1) # 资产和现金平均初始资金
    g_rate=0.02
    g=np.zeros([N_t,1])
    g[0:int(reduced_node[0]),0]=initial_wealth*(1+g_rate)*np.ones(int(reduced_node[0]))
    for t in range(stage-1):
        g[int(np.sum(reduced_node[0:t+1])):int(np.sum(reduced_node[0:t+2])),0]=initial_wealth*(1+(t+2)*g_rate)*np.ones(int(reduced_node[t+1]))
    ####################### 优化模型
    model = grb.Model()
    # 定义矩阵变量时候，需要用model.addVars不是model.addVar
    y0 = model.addVars(1, 1, vtype=grb.GRB.CONTINUOUS,lb=0)
    x0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b0 = model.addVars(1,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    # xr0= model.addVar(1, d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    #############################################################
    y = model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS,lb=0)
    x = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    b = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)
    s = model.addVars(N_t,d-2,vtype=grb.GRB.CONTINUOUS,lb=0)

    # xr=model.addVars(stage-1,d-2,leaf_node,vtype=grb.GRB.CONTINUOUS,lb=0)
    ######### cvar variables#############
    z=model.addVars(N_t,1,vtype=grb.GRB.CONTINUOUS)
    ################# constraints
    for i in range(d-2):
        model.addConstr(x0[0,i]==x0_+b0[0,i]-s0[0,i])
        model.addConstr(s0[0,i]<=x0_)
    model.addConstr(y0[0,0]==x0_-(1+c_b)*b0.sum(0,'*')+(1-c_s)*s0.sum(0,'*'))
    ############ t=1
    for i in range(int(reduced_node[0])):
        for j in range(d-2):
            model.addConstr(x[i,j]==x0[0,j]*cell_tree[0,0][i,j]+b[i,j]-s[i,j])
            model.addConstr(s[i,j]<=x0[0,j]*cell_tree[0,0][i,j])
        model.addConstr(y[i,0]==y0[0,0]*(1+interest_rate)-(1+c_b)*b.sum(i,'*')+(1-c_s)*s.sum(i,'*'))
        model.addConstr(z[i,0]==grb.quicksum(x0[0,k]*cell_tree[0,0][i,k] for k in range(d-2))+y0[0,0]*(1+interest_rate))
    ############# t=2
    for t in range(1,stage):
        for i in range(int(reduced_node[t])):
            for j in range(d-2):
                if t==1:
                    model.addConstr(x[int(reduced_node[0])+i,j]==x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(reduced_node[0])+i,j]-s[int(reduced_node[0])+i,j])
                    model.addConstr(s[int(reduced_node[0])+i,j]<=x[n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
                else:
                    model.addConstr(x[int(np.sum(reduced_node[0:t]))+i,j]==x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j]+b[int(np.sum(reduced_node[0:t]))+i,j]-s[int(np.sum(reduced_node[0:t]))+i,j])
                    model.addConstr(s[int(np.sum(reduced_node[0:t]))+i,j]<=x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],j]*cell_tree[0,t][i,j])
            if t==1:
                model.addConstr(y[int(reduced_node[0])+i,0]==y[n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(int(reduced_node[0])+i,'*')-(1+c_b)*b.sum(int(reduced_node[0])+i,'*'))
                model.addConstr(z[int(reduced_node[0])+i,0]==grb.quicksum(x[n_br_father[0,t][i,0],k]*cell_tree[0,t][i,(d-2)*(t-1)+k] for k in range(d-2))+y[n_br_father[0,t][i,0],0]*(1+interest_rate))
            else:
                model.addConstr(y[int(np.sum(reduced_node[0:t]))+i,0]==y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate)+(1-c_s)*s.sum(np.sum(reduced_node[0:t]+i),'*')-(1+c_b)*b.sum(int(np.sum(reduced_node[0:t])+i),'*'))
                model.addConstr(z[int(np.sum(reduced_node[0:t]))+i,0]==grb.quicksum(x[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],k]*cell_tree[0,t][i,k] for k in range(d-2))+y[int(np.sum(reduced_node[0:t-1]))+n_br_father[0,t][i,0],0]*(1+interest_rate))

    # for i in range(leaf_node):
    #     model.addConstr(g[np.sum(reduced_node[0:stage-1])+i,0]-z[np.sum(reduced_node[0:stage-1])+i,0]<=zc+f[np.sum(reduced_node[0:stage-1])+i,0])

    # model.addConstr(cvar==zc+(1/(1-alpha))*grb.quicksum(f[np.sum(reduced_node[0:stage-1])+k,0]*cell_tree[0,-1][k,-3] for k in range(leaf_node)))
    gamma1=1
    gamma2=700
    me_v = model.addVar(vtype=grb.GRB.CONTINUOUS)
    model.addConstr(me_v ==grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)))
    model.setObjective(me_v-grb.quicksum((cell_tree[0,-1][i,-2]*((z[int(np.sum(reduced_node[0:stage-1]))+i, 0]-me_v)*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0]-me_v))) for i in range(leaf_node)), sense=grb.GRB.MAXIMIZE)

    # model.setObjective(grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)) - grb.quicksum((cell_tree[0,-1][i,-2]*((z[int(np.sum(reduced_node[0:stage - 1]))+i,0]-grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i, 0])) for i in range(leaf_node)))*(z[int(np.sum(reduced_node[0:stage - 1])) + i, 0] - grb.quicksum((cell_tree[0,-1][i,-2]*(z[int(np.sum(reduced_node[0:stage-1]))+i,0])) for i in range(leaf_node))))) for i in range(leaf_node)),sense=grb.GRB.MAXIMIZE)
    model.optimize()
    obj = model.ObjVal
    x0_values = np.zeros(d - 1)
    x0_values[0] = y0[0, 0].x
    for i in range(d - 2):
        x0_values[1 + i] = x0[0, i].x
    obj = model.ObjVal
    return obj, x0_values

