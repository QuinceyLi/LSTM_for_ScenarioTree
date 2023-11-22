function [residuals, mu] = ARMAeq(theta, data, spec)

[T,n] = size(data);
if n > 1
    error('the data should be univariate (one column)')
end
ar = spec.ar;
ma=spec.ma;
gp=spec.gp;
gq=spec.gq;
m=max([ar,ma,gp,gq]);
% covs = ones(T,m);
% for i = 2:m
%     covs(:,i) = [zeros(i-1,1); data(1:end-i+1)];
% end

%mu = sum(repmat(theta',[T,1]).*covs,2);


mu = zeros(T,1);
mu(1:m) = theta(1);
for t = m+1:T
   mu(t) = theta'*[1; data(t-(1:ar)); data(t-(1:ma))-mu(t-(1:ma))];%; xy(t,:)'*ones((isscalar(x) < 1))];
end

residuals = data - mu;

% for t = (m+1):T; 
%    mu(t,1) = parameters(1:1+z)'*[1; data(t-(1:ar)); data(t-(1:ma))-mu(t-(1:ma),1); xy(t,:)'*ones((isscalar(x) < 1))];
% end
