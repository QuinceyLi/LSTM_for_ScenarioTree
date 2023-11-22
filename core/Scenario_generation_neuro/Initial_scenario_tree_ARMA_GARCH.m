clc
clear
t0=datetime("now");
%--------------------------------------------------------------------------
load  stock_data
stock=return_data;
statistics.mean=mean(stock);
statistics.var=cov(stock);
%stock=[factors4(1:end,:)];
[N,n_stock]=size(stock);
roll_step=238;
skip=4;
n_tree=floor(roll_step/skip);
% uv=100;
% lv=8;
%--------------------------------------------------------------------
stage0=3;
n_branch0=[1,15,8,5];
sum_branch0=cumprod(n_branch0);
sum_branch1=cumsum(sum_branch0)-ones(1,stage0+1);
scenario=prod(n_branch0);
N_start=N-roll_step;
Tree_mean=cell(1,roll_step);
Tree_sqvar=cell(1,roll_step);
for ik=1:1:n_tree
    n_branch=n_branch0;
    stage=stage0;
    % 这里注意如果len(stock) mod skip ==0，则会多一个情景树
    stock_history_data=stock((ik-1)*skip+1:N_start+(ik-1)*skip,:);
    [node_data,n_br,n_node,time,tree_mean,tree_sqvar,parsM,parsC]=b_scenario_generation_ARMA_GARCH_no_kmeans(stock_history_data,n_branch,stage,scenario);
    %[node_data,n_br,n_node,time,tree_mean,tree_sqvar,parsM,finaldata1]=b_scenario_generation_ARMA_GARCH_kmeans(stock_history_data,n_branch,stage,scenario)
    Tree_mean{1,ik}=tree_mean;
    Tree_sqvar{1,ik}=tree_sqvar;
    Node_data{1,ik}=node_data;
    ik
end
disp(datetime("now")-t0)

%%
tree=cell(n_tree,stage0);
for j=1:1:n_tree
    node_data = Node_data{1,j};
    for i=1:1:stage0
        tree{j,i}=node_data(sum_branch1(i)+1:sum_branch1(i+1),:);
    end
end
%% 
% current_time=datestr(now,'yyyy-mm-dd_HH:MM:SS');
% current_time=datetime("now");

% 获取当前日期和时间  
timestamp = datestr(now);   
% 用当前时间命名文件  
filename1 = sprintf('ag_tree_%s.mat', timestamp);
filename1=strrep(filename1,':','_');
save(['C:\Users\99591\Desktop\research\ScenarioTree\Bayesian\data\',filename1],'tree');
