function [node_data,n_br,n_node,time,tree_mean,tree_sqvar,parsM,parsC,finaldata1]=b_scenario_generation_tDCCcopula_sket_no_kmeans_moment_matching(stock_history_data,n_branch,stage,scenario,statistics)
t1=clock;
stock=stock_history_data;
data=stock;
NN=scenario;%���龰��
n_node_stage = cumprod(n_branch(2:end));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n_sample(1)=NN;
% for t=2:stage
%     n_sample(t)=fix(NN/prod(n_branch(1:t)));%n_branch=[1,20,10,10,10] n_sample=[20000,1000,100,10]
% end
%%%%%%%%%%%%%%%����k-means  �ȸ��ʵĲ���%%%%%%%%%%%%%%%%%%%%%%
n_sample=n_branch(2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sample_total,n_stock] = size(stock);


finaldata=zeros(n_sample(1),n_stock);

ar=1;ma=0;gp=1;gq=1;
spec = Mixmodelspec_garch(data,ar,ma,gp,gq);
[parsM, LogL, evalmodel, GradHess, varargout] = MixfitModel(spec, data);
[residuals,ht]=calculate_rh(data,parsM,spec);
udata=varargout;ֵ
stdRes = residuals./sqrt(ht);
%[parsC,LogLC,evalC]=b_tDCC_copula_estimate(udata,stdRes,uv,lv);%copula��������
spec = Mixmodelspec_copula(udata);
[parsC, LogL, evalmodel, GradHess] = MixfitModel(spec, udata);

% ԭʼ����������ת��Ϊ��׼�в�
[T,N]=size(udata);
% %nu=parsC(1);
% trdata=stdRes;
% [vt,Rt, veclRt]=b_DCCeq(parsC,trdata,udata,uv,lv);
% R=Rt(:,:,T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
vt=nueq(parsC(1:2),udata);
nu=vt(T);
% trdata=data;
%     for i=1:T
%           trdata(i,:)=tinv(data(i,:),nu(i));
%     end
trdata=stdRes;
[Rt, veclRt,Qt]=(parsC(3:end),trdata,udata);
R=Rt(:,:,T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:n_stock
    R(i,i)=1;
end
%nu=vt(1,T);
A=chol(R);
u=zeros(n_sample(1),N);
ee=u;
residuals=zeros(size(data));
ht=residuals;
% ����
solution_state='Infeasible';%��Ϊ'infeasible', ��moment-matching�Ľ��Ϊinfeasible, �����³���
while strcmp(solution_state,'Solved')~=1
    u(:,:)=copularnd('t',R,nu,n_sample(1));
%end
% ��׼�в�����ת��Ϊ����������
for j=1:N
    [residuals(:,j),~] =ARMAeq_mixed(parsM(1:2,j), data(:,j));
    ht(:,j)=VarEq_mixed(parsM(3:5,j), residuals(:,j));
end
ht=[ht;zeros(1,N)];
for j=1:N
    for i=1:n_sample(1)
%         options = optimoptions('fsolve','Algorithm','levenberg-marquardt');
%         ee(i,j)=fsolve(@(x)parsM(6,j)*normcdf(x,0,parsM(7,j))+(1-parsM(6,j))*normcdf(x,0,parsM(8,j))-u(i,j),0,options); %�����̬�ֲ��ķ�����
            ee(i,j)=skewtdis_inv(u(i,j), parsM(end-1,j), parsM(end,j));
    end  
    ht(T+1,j)=parsM(3,j)+parsM(4,j)*residuals(T,j)^2+parsM(5,j)*ht(T,j);    
    finaldata(:,j)=parsM(1,j)+parsM(2,j)*data(T,j)+sqrt(ht(T+1,j))*ee(:,j);
end
n_node = sum(cumprod(n_branch(2:end)));
node_data = zeros(n_node, n_stock+2);

finaldata1=finaldata;
node_data(1:n_branch(2),1:n_stock) = finaldata;

% �������õ��ĸ���
% pr = zeros(n_branch(2),1);
% for i = 1:n_branch(2)
%     pr(i,1) = 1/n_sample(1);
% end
% node_data(1:n_branch(2),end-1) = pr;
% node_data(1:n_branch(2),end) = pr;
%%��ƥ��
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
    solution_state='Infeasible';%��Ϊ'infeasible', ��moment-matching�Ľ��Ϊinfeasible, �����³���
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
        
        % ԭʼ���������ݺ���������ת��Ϊ��׼�в�����
        [T,N]=size(data);
        udata=zeros(size(data));
        residuals=zeros(size(data));
        ht=zeros(size(data));
        stdRes=zeros(size(data));
        for j=1:N
            [residuals(:,j),~] =  ARMAeq_mixed(parsM(1:2,j), data(:,j));
            ht(:,j)=VarEq_mixed(parsM(3:5,j), residuals(:,j));
            stdRes(:,j)= residuals(:,j)./sqrt(ht(:,j));
            for i=1:T
                udata(i,j)=skewtdis_cdf(stdRes(i,j),parsM(end-1,j),parsM(end,j));
            end
        end        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%20200610
vt=nueq(parsC(1:2),udata);
nu=vt(T);
trdata=stdRes;
[Rt, veclRt,Qt]=DCCeq(parsC(3:end),trdata,udata);
R=Rt(:,:,T);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:n_stock
             R(i,i)=1;
        end
        A=chol(R);
        u=zeros(n_sample(t),N);
        ee=u;
        % ����
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         for i=1:n_sample(t)
%             xx=randn(N,1);
%             x=A*xx;
%             x=x';
%             xi=chi2rnd(nu);
%             x=x./sqrt(xi/nu);
%             u(i,:)=tcdf(x,nu);
%         end
%         % ������׼���вԭ������������
%         ht=[ht;zeros(1,N)];
%         for j=1:N
%             for i=1:n_sample(t)
%                 options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','none');
%                 ee(i,j)=fsolve(@(x)parsM(6,j)*normcdf(x,0,parsM(7,j))+(1-parsM(6,j))*normcdf(x,0,parsM(8,j))-u(i,j),0,options);
%             end
%             ht(T+1,j)=parsM(3,j)+parsM(4,j)*residuals(T,j)^2+parsM(5,j)*ht(T,j);
%             finaldata(:,j)=parsM(1,j)+parsM(2,j)*data(T,j)+sqrt(ht(T+1,j))*ee(:,j);
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%20200609��ϸ�˹ת��ζƫt�ֲ����龰�����ɲ���code%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       u(:,:)=copularnd('t',R,nu,n_sample(t));
    ht=[ht;zeros(1,N)];
for j=1:N
    for i=1:n_sample(t)
            ee(i,j)=skewtdis_inv(u(i,j), parsM(end-1,j), parsM(end,j));
    end  
    ht(T+1,j)=parsM(3,j)+parsM(4,j)*residuals(T,j)^2+parsM(5,j)*ht(T,j);
    finaldata(:,j)=parsM(1,j)+parsM(2,j)*data(T,j)+sqrt(ht(T+1,j))*ee(:,j);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),1:n_stock) = finaldata;
        
        % �������õ��ĸ���
        
%         pr = zeros(n_branch(t+1),1);
%         for i = 1:n_branch(t+1)
%             pr(i,1) = 1/n_sample(t);
%         end
%         node_data(cur_node+(kt-1)*n_branch(t+1)+1:cur_node+kt*n_branch(t+1),end) = pr;
    end
    %��ƥ��
        
        Pro_last=node_data(pre_node+1:cur_node,end-1);     
        node_data_return=node_data(sum(n_node_stage(1:t-1))+1:sum(n_node_stage(1:t)),1:end-2);
        [Pro,solution_state] =momentmatching_single_period_cvx(node_data_return,statistics,Pro_last,n_branch(t+1));
       
    end
      node_data(sum(n_node_stage(1:t-1))+1:sum(n_node_stage(1:t)),end-1)=Pro;  
end
%t2 = clock;
%%%% �ӽڵ�ĸ��ʣ������������ʣ�

for t = 2:stage
    pre_node = sum(cumprod(n_branch(2:t-1)));
    cur_node = sum(cumprod(n_branch(2:t)));
    n_cur_node = prod(n_branch(1:t+1));
    
    for kt = 1:n_cur_node
        par_node = floor((n_branch(t+1)+kt-1)/n_branch(t+1));
        node_data(cur_node+kt,end) = node_data(cur_node+kt,end-1)/node_data(pre_node+par_node,end-1);
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

