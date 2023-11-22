function [node_data,n_br,n_node,time,tree_mean,tree_sqvar,parsM,finaldata1]=b_scenario_generation_ARMA_GARCH_no_kmeans_MM(stock_history_data,n_branch,stage,scenario,statistics)
t1=clock;
stock=stock_history_data;
data=stock;%
NN=scenario;%
n_node_stage = cumprod(n_branch(2:end));
%%%%%%%%%%%%%%%????k-means  ?????????%%%%%%%%%%%%%%%%%%%%%%
n_sample=n_branch(2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sample_total,n_stock] = size(stock);

finaldata=zeros(n_sample(1),n_stock);
t3=clock;

%[parsM,LogLM,evalM,udata,stdRes]=b_garch_sket_estimate(data);%arma-garch????????
ar=1;ma=0;gp=1;gq=1;
spec = Mixmodelspec_garch(data,ar,ma,gp,gq);
[parsM, LogL, evalmodel, GradHess, varargout] = MixfitModel(spec, data);
[residuals,ht]=calculate_rh(data,parsM,spec);
udata=varargout;%??stdRes
stdRes = residuals./sqrt(ht);% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%????stdres???? ?????
[T,N]=size(udata);
mu_res=mean(stdRes); 
std_res=cov(stdRes);

solution_state='Infeasible';%?1?7?1?7?0?2'infeasible', ?1?7?1?7moment-matching?1?7?0?5?1?7?1?7?0?2infeasible, ?1?7?1?7?1?7?1?7?1?7?0?6?1?7?1?7?1?7
while strcmp(solution_state,'Solved')~=1
    ee= mvnrnd(mu_res,std_res,n_sample(1));

for j=1:N
    [residuals(:,j),~] =ARMAeq_mixed(parsM(1:2,j), data(:,j));
    ht(:,j)=VarEq_mixed(parsM(3:5,j), residuals(:,j));
end
ht=[ht;zeros(1,N)];
for j=1:N
    ht(T+1,j)=parsM(3,j)+parsM(4,j)*residuals(T,j)^2+parsM(5,j)*ht(T,j);    
    finaldata(:,j)=parsM(1,j)+parsM(2,j)*data(T,j)+sqrt(ht(T+1,j))*ee(:,j);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_node = sum(cumprod(n_branch(2:end)));
node_data = zeros(n_node, n_stock+2);

finaldata1=finaldata;
% ???????????
% opts = statset('Display','final','MaxIter',1000);
% [I, C] = kmeans(finaldata, n_branch(2),'Options',opts,'emptyaction','singleton');
node_data(1:n_branch(2),1:n_stock) = finaldata;

[Pro,solution_state] =momentmatching_single_period_cvx(finaldata,statistics,1,n_branch(2));
node_data(1:n_branch(2),end-1) = Pro;
node_data(1:n_branch(2),end) = Pro;
end

for t = 2:stage
    temp_data = zeros(t-1, n_stock);
%     u_temp_data=temp_data;
    pre_node = sum(cumprod(n_branch(2:t-1)));
    n_cur_node = prod(n_branch(1:t));
    cur_node = pre_node + n_cur_node;

    solution_state='Infeasible';
    while strcmp(solution_state,'Solved')~=1

    for kt = 1:n_cur_node
        temp_data(end,:) = node_data(pre_node+kt,1:n_stock);
%         u_temp_data(end,:)=u_node_data(pre_node+kt,1:n_stock);
        temp_counter = 0;
        par_k = kt;
        for tt = t:-1:3
            temp_counter = temp_counter + 1;
            par_k = floor((n_branch(tt)+par_k-1)/n_branch(tt));
            temp_data(end-temp_counter,:) = node_data(sum(cumprod(n_branch(2:tt-2)))+par_k, 1:n_stock);                     
        end
%         u1=[u;u_temp_data];
        stock1=[stock;temp_data];
        data=stock1;
        finaldata=zeros(n_sample(t),n_stock);
        [T,N]=size(data);
        udata=zeros(size(data));
        residuals=zeros(size(data));
        ht=zeros(size(data));
        stdRes=zeros(size(data));
        for j=1:N
            [residuals(:,j),~] =  ARMAeq_mixed(parsM(1:2,j), data(:,j));
            ht(:,j)=VarEq_mixed(parsM(3:5,j), residuals(:,j));
            stdRes(:,j)= residuals(:,j)./sqrt(ht(:,j));
        end        
ee= mvnrnd(mu_res,std_res,n_sample(t));
ht=[ht;zeros(1,N)];
for j=1:N 
    ht(T+1,j)=parsM(3,j)+parsM(4,j)*residuals(T,j)^2+parsM(5,j)*ht(T,j);
    finaldata(:,j)=parsM(1,j)+parsM(2,j)*data(T,j)+sqrt(ht(T+1,j))*ee(:,j);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),1:n_stock) = finaldata;
    end
        Pro_last=node_data(pre_node+1:cur_node,end-1);     
        node_data_return=node_data(sum(n_node_stage(1:t-1))+1:sum(n_node_stage(1:t)),1:end-2);
        [Pro,solution_state] =momentmatching_single_period_cvx(node_data_return,statistics,Pro_last,n_branch(t+1));
    end
      node_data(sum(n_node_stage(1:t-1))+1:sum(n_node_stage(1:t)),end-1)=Pro;  
end

for t = 2:stage
    pre_node = sum(cumprod(n_branch(2:t-1)));
    cur_node = sum(cumprod(n_branch(2:t)));
    n_cur_node = prod(n_branch(1:t+1));
    
    for kt = 1:n_cur_node
        par_node = floor((n_branch(t+1)+kt-1)/n_branch(t+1));
        node_data(cur_node+kt,end) = node_data(cur_node+kt,end-1)/node_data(pre_node+par_node,end-1);
    end
end

%????????��???
tree_mean = zeros(stage, n_stock);

tree_sqvar=tree_mean;
for t = 1:stage
    row_index = [sum(cumprod(n_branch(2:t+1)))-prod(n_branch(2:t+1))+1:sum(cumprod(n_branch(2:t+1)))];
    tree_mean(t,:) = (node_data(row_index,1:n_stock)'*node_data(row_index,end-1))';
    node_data_i=node_data(row_index,1:n_stock);
    for i=1:n_stock
        node_data_i(:,i)=(node_data_i(:,i)-tree_mean(t,i)).^2;
    end
    tree_sqvar(t,:)=sqrt((node_data_i(:,1:n_stock)'*node_data(row_index,end-1))');
end
% plot the scenario tree for each stock
% ???????????��
single_sce = zeros(stage, n_stock, prod(n_branch));
for i = 1:prod(n_branch)
    single_sce(t, :, i) = node_data(sum(cumprod(n_branch(2:t)))+i, 1:n_stock);
    par_node = floor((i+n_branch(t+1)-1)/n_branch(t+1));
    for j = t-1:-1:1
        single_sce(j, :, i) = node_data(sum(cumprod(n_branch(2:j)))+par_node, 1:n_stock);
        par_node = floor((par_node+n_branch(j+1)-1)/n_branch(j+1));
    end
end


for t = 1:stage
    n_br{1,t} = zeros(prod(n_branch(1:t)),1)+ n_branch(1,t+1);
end
n_node = cumprod(n_branch(2:end));
%node_data(:,1:n_stock)=exp(node_data(:,1:n_stock));
t2=clock;
time=etime(t2,t1);

