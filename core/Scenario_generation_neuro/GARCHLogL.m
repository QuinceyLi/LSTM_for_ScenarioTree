function [LogL, LL, ht, stdRes, udata]=GARCHLogL(theta,data,spec,solver)
% the marginal GARCH likelihood
% INPUTS
% theta:        vector of parameters
% data:         column vector of unfiltered data
% spec:         structure that contains the model specifications

% OUTPUTS
% ll:           the log likelihood value at the optimum
% ht:           vector of conditional variances
% stdRes:       vector of the iid standardized residuals
% udata:        vector of the transformed iid residuals
% -----------------------------------------------------------------------
% author: manthos Vogiatzoglou, UoM, 2010
% -----------------------------------------------------------------------
if nargin == 3
    solver = 'fmincon';
end

if strcmp(solver,'fminunc')==1
    % the input vector theta should be unconstrained
    % but the log likelihood accepts only constrained parameters
    if strcmp(spec.purpose,'fitCopulaGARCH')==1
        spec.purpose = 'fitGARCH';
        theta = RescaleParameters(theta, 1, spec);
        spec.purpose = 'fitCopulaGARCH';
    else
        theta = RescaleParameters(theta, 1, spec);
    end
    % thus the initial theta is transformed to the corresponding
    % constrained theta
end
%theta % you might want to see this if estimation crushes
mp = theta(1:1+spec.ar+spec.ma);    
residuals =  ARMAeq(mp, data, spec);
phi = theta(1+spec.ar+spec.ma+1:end);
if strcmp(spec.VarEq,'GARCH(1,1)')==1
    vp = phi(1:spec.gp+spec.gq+1); dp = phi(end-1:end);
else
    vp = phi(1:4); dp = phi(5:end);
end
ht=VarEq(vp, residuals, spec);
stdRes = residuals./sqrt(ht);
if strcmp(spec.distr,'Gaussian')==1
    LL=.5*log(2*pi)+.5*log(ht)+.5*(stdRes.^2);
elseif strcmp(spec.distr,'T')==1
    LL=-gammaln(.5*(dp+1))+gammaln(.5*dp)+.5*log(pi*(dp-2))+.5*log(ht)+.5*(dp+1)*log(1+stdRes.^2/(dp-2));
elseif strcmp(spec.distr,'SkewT')==1
    nu = dp(1); lambda = dp(2);
    %ll = .5*sum(log(ht)) + skewtdis_LL([nu;lambda], stdRes);
    [xx, LL1] = skewtLL([nu;lambda], stdRes);
    LL = .5*log(ht) - LL1;
end
LogL=sum(LL);
% -------------------------------------------------------------------------

if strcmp(spec.PIT,'CML')==1
    udata = empiricalCDF(stdRes);
elseif strcmp(spec.PIT,'IFM')==1 && strcmp(spec.distr,'Gaussian')==1
    %udata = normcdf(stdRes);% 原始 标准正太
    udata = normcdf(stdRes,mean(stdRes),std(stdRes));
elseif strcmp(spec.PIT,'IFM')==1 && strcmp(spec.distr,'T')==1
    udata = tcdf(stdRes,dp);
elseif strcmp(spec.PIT,'IFM')==1 && strcmp(spec.distr,'SkewT')==1
   
    udata = skewtdis_cdf(stdRes,nu,lambda);
end
    for i=1:size(data,1)
        if udata(i)>.99999
            udata(i) = .99999;
        elseif udata(i)<10^-9;
            udata(i) = 10^-9;
        end
    end
