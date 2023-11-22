function ht=VarEq(theta, residuals, garchSpec)
% calculates the conditional variance for univariate data
% only. Two models are supported:
% 1) The GARCH(1,1) model of Bollerslev
%                      h(t) = w + a*residuals(t-1)^2 + b*h(t-1)
% 2) The GJR(1,1) model of
%        h(t) = w + a*residuals(t-1)^2 + b*h(t-1) + c*(residuals(t-1)<0)^2
% INPUTS:
% theta:            vector of parameters (w; a; b)or (w;a;b;c)
% residuals:        iid, zero mean data
% GarchSpec:        Structure that contains the model specifications
% OUTPUT:
% ht:               The conditional variance
% -----------------------------------------------------------------------
% author: Vogiatzoglou Manthos, UoM, 2010
% contact at: vogia@yahoo.com
% ------------------------------------------------------------------------
[T,n]=size(residuals);
if n~=1
    error('this is for univariate data only')
end
h0=var(residuals);
ht=zeros(T,1); ht(1)=h0;
if strcmp(garchSpec.VarEq,'GARCH(1,1)')==1
    for i=2:T
        ht(i)=theta(1)+theta(2)*residuals(i-1)^2+theta(3)*ht(i-1);
    end
elseif strcmp(garchSpec.VarEq,'GJR(1,1)')==1
    for i=2:T
        if residuals(i-1)>=0
            ht(i)=theta(1)+theta(2)*residuals(i-1)^2+theta(3)*ht(i-1); 
        else
            ht(i)=theta(1)+theta(2)*residuals(i-1)^2+theta(3)*ht(i-1)+theta(4)*residuals(i-1)^2;
        end
    end
elseif strcmp(garchSpec.VarEq,'constant variance')==1
    ht=theta(1);
end

