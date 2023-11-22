
clc
clear
%--------------------------------------------------------------------------
load stock_data.mat  
stock=return_data+1;
[N,n_stock]=size(stock);
roll_step=238;
%--------------------------------------------------------------------------
stage0=4;
n_branch0=[1,15,10,5,5];
scenario=prod(n_branch0);
N_start=N-roll_step;
Tree_mean=cell(1,roll_step);
Tree_sqvar=cell(1,roll_step);

for ik=1:1
    n=mod(ik-1,stage0);
    n_branch=n_branch0(1:end-n);
    stage=stage0-n;
    stock_history_data=stock(ik:N_start+ik-1,:);
    riskless_return=log(1.0005);
    [node_data,n_br,n_node,time,tree_mean,tree_sqvar]=scenario_generation_varmgarch(stock_history_data,n_branch,stage,riskless_return);  
    Tree_mean{1,ik}=tree_mean;
    Tree_sqvar{1,ik}=tree_sqvar;
    node_data(1:n_node(1),end-1)=repmat(1/n_node(1),n_node(1),1);
    for t=2:stage0
        node_data(sum(n_node(1:t-1))+1:sum(n_node(1:t)),end-1)=repmat(1/n_node(t),n_node(t),1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Node_data{1,ik}=node_data;
    root_node=stock_history_data(end,:);
end


%save initial_fantree_10times_201010_cell_tree_no_arbitrage initial_cell_tree_all root_node n_branch 

