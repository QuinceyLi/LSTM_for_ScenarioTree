function [residuals,ht,mu]=calculate_rh(data,parameters,spec)
[T,n_stock]=size(data);
parsM=parameters;
% residuals=data;
residuals=zeros(size(data));
ht=data;
mu=data;
for j=1:n_stock
    %[residuals(:,j),mu(:,j)] =  ARMAeq(parsM(1:1+spec.ar+spec.ma,j), data(:,j), spec);
    [residuals(:,j),mu(:,j)] =ARMAeq_mixed(parsM(1:1+spec.ar+spec.ma,j), data(:,j));
    ht(:,j)=VarEq(parsM(1+spec.ar+spec.ma+1:end,j), residuals(:,j), spec);
end

    