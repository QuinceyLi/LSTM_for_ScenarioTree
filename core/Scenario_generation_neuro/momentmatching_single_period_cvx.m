%%%%%%%%%%%%%%%%%%%%% 用cvx求解能动大规模线性规划问题代码
function [Pro,solution_state] =momentmatching_single_period_cvx(sample,statistics,pro_last,nnn_branch)
%Pro为匹配后得到的概率

%Pro_last输入的节点的所有父节点对应的概率
mu=statistics.mean;
sigma=statistics.var;
n=size(sigma,1);
m=size(sample,1);
mm=size(pro_last,1);
%采样
sample=lhsnorm(mu,sigma,m);%拉丁超立方抽样 m*n
%B=mvnrnd(mu,sigma,m);%随机采样
sample(isnan(sample)) = 0;
C=sample';
%%%%%%%%%%%%%%%%%%%%%%%%%moment matching model
f=ones(1,m+n*(n+1));
ff=zeros(1,m);
f(1,1:m)=ff;
gg=ones(1,m);

cvx_begin %sdp   y(:,j,:)*Phi
cvx_solver mosek
    variable x(m+n*(1+n),1) 
    minimize(f*x )
    subject to
        for i=1:n
            C(i,:)*x(1:m,1)==mu(i);
        end
        for j=1:n
            for k=j:n
                C(j,:).*C(k,:)*x(1:m,1)+x(m+(2*n-j+2)*(j-1)+2*(k-j)+1,1)-x(m+(2*n-j+2)*(j-1)+2*(k-j)+2,1)==sigma(j,k);
            end
        end
        for k=1:mm
            sum(x(1+nnn_branch*(k-1):nnn_branch+nnn_branch*(k-1),1))==pro_last(k);
        end
        x(1:m)>=min(pro_last)/nnn_branch/10;
        x(m+1:end)>=0;
cvx_end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cvx_status :Solved /Infeasible
 Pro=x(1:m);
 solution_state=cvx_status;
end

