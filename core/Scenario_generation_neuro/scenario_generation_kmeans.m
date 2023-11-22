function [node_data,n_br]=scenario_generation_kmeans(simulated_tree_data,n_branch)
   %算法3
   
   [stage n_assets n_path] = size(simulated_tree_data);
Index_set(1:sum(cumprod(n_branch)), 1) = {[]}; % 每类的指标集
Means_set(1:sum(cumprod(n_branch)), 1) = {[]};
Index_set(1, 1) = {[1:n_path]'};

for t = 1:stage
    node_index = sum(cumprod(n_branch(1:t)));%For example:(cumprod([1 10 9 8])=[1    10    90   720]
    for k = 1:prod(n_branch(1:t))%If A is a vector, prod(A) returns the product of the elements.For example:prod([1 10 90])=900
        data_matrix = [];
        temp_index = node_index-prod(n_branch(1:t))+k;
        for i = 1:size(Index_set{temp_index,1}, 1)   %size(b,1)返回b的行数；
            data_matrix = [data_matrix; simulated_tree_data(t,:,Index_set{temp_index,1}(i))];
        end
        [IDX, C,SumD] = kmeans(data_matrix, n_branch(t+1),'Maxiter',1000);%SumD是类间所有点与该质心点距离之和%IDX存贮的是每个点的聚类标号；C存贮的树聚类中心
        kk=0;
        CC=zeros(1,n_branch(t+1));
        for i = 1:n_branch(t+1)
            Index_set(node_index+(k-1)*n_branch(t+1)+i, 1) = {Index_set{temp_index,1}((IDX == i))};%Index_set（i，1）存贮的是聚类后第i个节点包含的扇形情景树的情景标号的集合；
            kk=kk+1;
            CC(kk)=length(Index_set{node_index+(k-1)*n_branch(t+1)+i, 1});
            Means_set(node_index+(k-1)*n_branch(t+1)+i, 1) = {C(i,:)};%Means_set（i，1）存贮的是聚类后第i个节点的中心；
        end
        if t<stage
            while min(CC)<n_branch(t+2)
                [IDX, C,SumD] = kmeans(data_matrix, n_branch(t+1),'Maxiter',1000);%SumD是类间所有点与该质心点距离之和%IDX存贮的是每个点的聚类标号；C存贮的树聚类中心
                kk=0;
                CC=zeros(1,n_branch(t+1));
                for i = 1:n_branch(t+1)
                    Index_set(node_index+(k-1)*n_branch(t+1)+i, 1) = {Index_set{temp_index,1}((IDX == i))};%Index_set（i，1）存贮的是聚类后第i个节点包含的扇形情景树的情景标号的集合；
                    kk=kk+1;
                    CC(kk)=length(Index_set{node_index+(k-1)*n_branch(t+1)+i, 1});
                    Means_set(node_index+(k-1)*n_branch(t+1)+i, 1) = {C(i,:)};%Means_set（i，1）存贮的是聚类后第i个节点的中心；
                end
            end
        end
        
    end
end
% 聚类形成的概率 Prob_set
for i = 1:sum(cumprod(n_branch))
    Prob_set(i,1) = size(Index_set{i,1}, 1)/n_path;
end

% 风扇型情景树上的各阶矩
original_means = zeros(stage, n_assets);
original_co_var = zeros(n_assets, n_assets, stage);
original_skewness = zeros(stage, n_assets);
original_kurtosis = zeros(stage, n_assets);
for t = 1:stage
    for i = 1:n_path
        original_means(t,:) = original_means(t,:)+simulated_tree_data(t,:,i);
    end
    original_means(t,:) = original_means(t,:)./n_path;
    for i = 1:n_path
        original_co_var(:,:,t) = original_co_var(:,:,t)+(simulated_tree_data(t,:,i)-original_means(t,:))'*(simulated_tree_data(t,:,i)-original_means(t,:));
        original_skewness(t,:) = original_skewness(t,:)+(simulated_tree_data(t,:,i)-original_means(t,:)).^3;
        original_kurtosis(t,:) = original_kurtosis(t,:)+(simulated_tree_data(t,:,i)-original_means(t,:)).^4;
    end
    original_co_var(:,:,t) = original_co_var(:,:,t)./(n_path-1);
    original_skewness(t,:) = original_skewness(t,:).*(n_path/((n_path-1)*(n_path-2)));
    original_kurtosis(t,:) = original_kurtosis(t,:).*(n_path*(n_path+1)/((n_path-1)*(n_path-2)*(n_path-3)));
end
% 聚类后的各阶矩
cluster_means = zeros(stage, n_assets);
cluster_co_var = zeros(n_assets, n_assets, stage);
cluster_skewness = zeros(stage, n_assets);
cluster_kurtosis = zeros(stage, n_assets);
for t = 1:stage
    node_index = sum(cumprod(n_branch(1:t)));
    for i = 1:prod(n_branch(1:t+1))
        cluster_means(t,:) = cluster_means(t,:)+Means_set{node_index+i,1}.*Prob_set(node_index+i);
    end
    for i = 1:prod(n_branch(1:t+1))
        cluster_co_var(:,:,t) = cluster_co_var(:,:,t)+(Means_set{node_index+i,1}-cluster_means(t,:))'*(Means_set{node_index+i,1}-cluster_means(t,:)).*Prob_set(node_index+i);
        cluster_skewness(t,:) = cluster_skewness(t,:)+(Means_set{node_index+i,1}-cluster_means(t,:)).^3.*Prob_set(node_index+i);
        cluster_kurtosis(t,:) = cluster_kurtosis(t,:)+(Means_set{node_index+i,1}-cluster_means(t,:)).^4.*Prob_set(node_index+i);
    end
end

% save Index_set.mat Index_set
% save Means_set.mat Means_set
% save Prob_set.mat Prob_set
% save cluster_properties.mat cluster_means cluster_co_var cluster_skewness cluster_kurtosis
% save original_properties.mat original_means original_co_var original_skewness original_kurtosis
% % 聚类所得情景树
node_data = zeros(sum(cumprod(n_branch(2:end))),n_assets);
for i = 1:sum(cumprod(n_branch(2:end)))
    node_data(i,1:n_assets) = exp(Means_set{i+1,:});
    node_data(i,n_assets+1) = length(Index_set{i+1,1})/n_path;
end

node_data(1:n_branch(2),n_assets+2) = node_data(1:n_branch(2),n_assets+1);
for t = 2:stage
    pre_node = sum(cumprod(n_branch(1:t-1)))-1;
    cur_node = prod(n_branch(1:t+1));
    for kt = 1:cur_node
        node_data(sum(cumprod(n_branch(1:t)))-1+kt,n_assets+2) = node_data(sum(cumprod(n_branch(1:t)))-1+kt,n_assets+1)/node_data(pre_node+floor((kt+n_branch(t+1)-1)/n_branch(t+1)),n_assets+1);
    end
end
   
   node_data = zeros(sum(cumprod(n_branch(2:end))),n_assets);
for i = 1:sum(cumprod(n_branch(2:end)))
    node_data(i,1:n_assets) = exp(Means_set{i+1,:});
    node_data(i,n_assets+1) = length(Index_set{i+1,1})/n_path;
end

node_data(1:n_branch(2),n_assets+2) = node_data(1:n_branch(2),n_assets+1);
for t = 2:stage
    pre_node = sum(cumprod(n_branch(1:t-1)))-1;
    cur_node = prod(n_branch(1:t+1));
    for kt = 1:cur_node
        node_data(sum(cumprod(n_branch(1:t)))-1+kt,n_assets+2) = node_data(sum(cumprod(n_branch(1:t)))-1+kt,n_assets+1)/node_data(pre_node+floor((kt+n_branch(t+1)-1)/n_branch(t+1)),n_assets+1);
    end
end

n_br(1,1:stage) = {[]};
n_br{1,1}(1,1) = n_branch(2);
for i=2:stage
    n_br{1,i}=ones(prod(n_branch(1:i)),1)*n_branch(i+1);
end



