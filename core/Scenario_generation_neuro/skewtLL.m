function [LogL, LL] = skewtLL(theta, x)

nu = theta(1);
lambda = theta(2);

T = size(x,1);
   
c = gamma((nu+1)/2)./(sqrt(pi*(nu-2)).*gamma(nu/2));
a = 4*lambda.*c.*((nu-2)./(nu-1));
b = sqrt(1 + 3*lambda.^2 - a.^2);

logc = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*(nu-2));
logb = 0.5*log(1 + 3*lambda.^2 - a.^2);
LL = zeros(size(x));

for i=1:T
    if x(i)<-a/b
        LL(i)   = logb + logc - (nu+1)/2.*log(1+1./(nu-2).*((b.*x(i)+a)./(1-lambda)).^2);
    else
        LL(i)   = logb + logc - (nu+1)/2.*log(1+1./(nu-2).*((b.*x(i)+a)./(1+lambda)).^2);
    end
end
LogL = -sum(LL);