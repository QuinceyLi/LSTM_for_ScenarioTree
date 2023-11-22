function ht=VarEq_mixed(theta, residuals)
[T,n]=size(residuals);
if n~=1
    error('this is for univariate data only')
end
h0=var(residuals);
ht=zeros(T,1); ht(1)=h0;
for i=2:T
    ht(i)=theta(1)+theta(2)*residuals(i-1)^2+theta(3)*ht(i-1);
end


