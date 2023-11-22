function [node_data,n_br,n_node,time,tree_mean,tree_sqvar]=scenario_generation_varmgarch_no_arbitrage(data,n_branch,stage,riskless_return)
t1=clock;
stock=data;
n_sample=n_branch(2:end);
[sample_total n_stock] = size(stock);
%�Իع����
r = 1;
% garchģ�Ͳ���
%     q = 2*ones(1,n_stock);
%     p = 2*ones(1,n_stock);
p = ones(1,n_stock);
q = ones(1,n_stock);
dccP = 1;
dccQ = 1;
%===================================�ع����===============================

bb = var_estimate(stock,r);

u=zeros(sample_total-r,n_stock);%�����в�
for j = r+1:sample_total
    u(j-1,:) = stock(j,:) - (bb*[1 stock(j-1,:)]')';
end


%===================================�ع��������=============================

%===============================�����ڼ۸�������ʼ���=======================
%����������Ƹ���

%=========================================dcc-MGARCH(p,q)����===================================================

% [garchparameters, loglikelihood, Ht, Qt, stdresid, likelihoods, stderrors, A,B, jointscores]=dcc_mvgarch(u , dccP , dccQ , p , q);
[garchparameters, loglikelihood, Ht, Qt] = dcc_mvgarch(u , dccP , dccQ , p , q);

% OUTPUTS:
%      parameters    = A vector of parameters estimated form the model of the form
%                          [GarchParams(1) GarchParams(2) ... GarchParams(k) DCCParams]
%                          where the garch parameters from each estimation are of the form
%                          [omega(i) alpha(i1) alpha(i2) ... alpha(ip(i)) beta(i1) beta(i2) ... beta(iq(i))]
%      loglikelihood = The log likelihood evaluated at the optimum
%      Ht            = A k by k by t array of conditional variances
%      likelihoods   = the estimated likelihoods t by 1
%      stderrors     = A length(parameters)^2 matrix of estimated correct standard errors
%      A             = The estimated A form the rebust standard errors
%      B             = The estimated B from the standard errors
%      scores        = The estimated scores of the likelihood t by length(parameters)

a = size(Ht);
for k = 1:a(3)
    for i = 1:n_stock
        aa(1,i) = Ht(i,i,k);
    end
    hmat(k,:) = aa(1,:);
end


%========================����������ļ۸��������============================

Qbar=cov(u./sqrt(hmat));

d = size(garchparameters,1);
dccparameters = garchparameters(d-dccP-dccQ+1:d,1);
garchparameters = garchparameters(1:d-dccP-dccQ,1);

EXITFLAG=0;
while EXITFLAG~=1
    finaldata=zeros(n_sample(1),n_stock);
    for i = 1:n_sample(1)
        finaldata(i,:)= dcc_sim_single_stage(n_stock,1,u,stock,Qbar,hmat,Qt,garchparameters,p,q,dccparameters,dccP,dccQ,r,bb);
    end
    n_node = sum(cumprod(n_branch(2:end)));
    %n_leaf = prod(n_branch);
    node_data = zeros(n_node, n_stock+2);
    u_node_data=node_data;
    % ���ɸ��ڵ�ķ�֦
%     opts = statset('Display','final','MaxIter',1000);
%     [I, C] = kmeans(finaldata, n_branch(2),'Options',opts);
    C=finaldata;
    cost_vec=zeros(n_branch(2),1);
    Aeq=C;
    Aeq=Aeq';
    Aeq=[Aeq;cost_vec'+1];
    beq = [riskless_return*ones(n_stock, 1); 1];
    options = optimset('Display','off');
    % [XX,FVAL,EXITFLAG] = linprog(cost_vec,[],[],Aeq,beq,cost_vec+0.0001,[],[],options); % lb = cost_vec
    [XX,FVAL,EXITFLAG] = linprog(cost_vec,[],[],Aeq,beq,cost_vec+0.0001,[],options) % lb = cost_vec
    
end

node_data(1:n_branch(2),1:n_stock) = C;
u_C=C;
for i=1:n_branch(2)
    u_C(i,:)=calculate_u(C(i,:),stock,bb);
end
u_node_data(1:n_branch(2),1:n_stock) = u_C;


% �������õ��ĸ���
pr = zeros(n_branch(2),1);
for i = 1:n_branch(2)
    pr(i,1) = 1/n_sample(1);
end
node_data(1:n_branch(2),end-1) = pr;
node_data(1:n_branch(2),end) = pr;


for t = 2:stage
    temp_data = zeros(t-1, n_stock);
    u_temp_data=temp_data;
    pre_node = sum(cumprod(n_branch(2:t-1)));
    n_cur_node = prod(n_branch(1:t));
    cur_node = pre_node + n_cur_node;
    for kt = 1:n_cur_node
        temp_data(end,:) = node_data(pre_node+kt,1:n_stock);
        u_temp_data(end,:)=u_node_data(pre_node+kt,1:n_stock);
        temp_counter = 0;
        par_k = kt;
        for tt = t:-1:3
            temp_counter = temp_counter + 1;
            par_k = floor((n_branch(tt)+par_k-1)/n_branch(tt));
            temp_data(end-temp_counter,:) = node_data(sum(cumprod(n_branch(2:tt-2)))+par_k, 1:n_stock);
            u_temp_data(end-temp_counter,:) = u_node_data(sum(cumprod(n_branch(2:tt-2)))+par_k, 1:n_stock);
            
        end
        u1=[u;u_temp_data];
        stock1=[stock;temp_data];
        EXITFLAG=0;
        while EXITFLAG~=1;
            finaldata=zeros(n_sample(t),n_stock);
            for i = 1:n_sample(t)
                finaldata(i,:)= dcc_sim_single_stage(n_stock,t,u1,stock1,Qbar,hmat,Qt,garchparameters,p,q,dccparameters,dccP,dccQ,r,bb);
            end
            
            %opts = statset('Display','final','MaxIter',1000);
            %[I, C] = kmeans(finaldata, n_branch(t+1),'Options',opts);
            C=finaldata;
            cost_vec=zeros(n_branch(t+1),1);
            Aeq=C;
            Aeq=Aeq';
            Aeq=[Aeq;cost_vec'+1];
            beq = [riskless_return*ones(n_stock, 1); 1];
            options = optimset('Display','off');
            % [XX,FVAL,EXITFLAG] = linprog(cost_vec,[],[],Aeq,beq,cost_vec+0.0001,[],[],options); % lb = cost_vec
            [XX,FVAL,EXITFLAG] = linprog(cost_vec,[],[],Aeq,beq,cost_vec+0.0001,[],options) % lb = cost_vec
            
        end
        node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),1:n_stock) = C;
        u_C=C;
        for i=1:n_branch(t+1)
            u_C(i,:)=calculate_u(C(i,:),stock1,bb);
        end
        u_node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),1:n_stock) = u_C;
        % �������õ��ĸ���
        
        pr = zeros(n_branch(t+1),1);
        for i = 1:n_branch(t+1)
            pr(i,1) = 1/n_sample(t);
        end
        node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),end) = pr;
    end
end
%t2 = clock;

for t = 2:stage
    pre_node = sum(cumprod(n_branch(2:t-1)));
    cur_node = sum(cumprod(n_branch(2:t)));
    n_cur_node = prod(n_branch(1:t+1));
    
    for kt = 1:n_cur_node
        par_node = floor((n_branch(t+1)+kt-1)/n_branch(t+1));
        node_data(cur_node+kt,end-1) = node_data(cur_node+kt,end)*node_data(pre_node+par_node,end-1);
    end
end

%�龰�����׶εľ�ֵ
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
% ��ÿ֧�龰�����洢
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

