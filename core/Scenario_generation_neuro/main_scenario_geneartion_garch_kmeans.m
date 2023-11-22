clear
clc
load stock_data
stage=3;
load simulated_tree_data_3_stage_10000scenario.mat
n_assets=12;
n_branch=[1 10 9 8];
 t1=clock;
[node_data,n_br]=scenario_generation_kmeans(simulated_tree_data,n_branch);
t2=clock;
time=etime(t2,t1);
save scenario_tree_3_stage_1098_g_k.mat

