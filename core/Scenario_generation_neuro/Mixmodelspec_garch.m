function spec = Mixmodelspec_garch(data,ar,ma,gp,gq)
% creates the structure that defines the model characteristics
[T,n] = size(data);
spec.size = n;
spec.purpose = 'fitGARCH';

aa=1;
spec.gp=gp;
spec.gq=gq;
if aa==1 || aa==3  % when GARCH models will be estimated
    
    %ab=input('input the lag - length of the AR terms in the mean equation and press enter:');
    %ab=1;
    spec.ar=ar;
    spec.ma=ma;
    spec.mtheta0=[mean(data(:,1));zeros(spec.ar,1);zeros(spec.ma,1)];
    ac=1;%menu('define the variance equation','GARCH(1,1)','GJR(1,1)');
    if ac==1
        spec.VarEq='GARCH(1,1)';
        spec.vparams=1+spec.gp+spec.gq;
        spec.vtheta0=[.02*var(data(:,1));.08/spec.gp*ones(spec.gp,1);.91/spec.gq*ones(spec.gq,1)];
    elseif ac==2
        spec.VarEq='GJR(1,1)';
        spec.vparams=4;
        spec.vtheta0=[.02*var(data(:,1));.05;.85;.15];
    end
    ad=3;%=menu('define the distribution of the residuals','Gaussian','T','skewT');
    if ad==1
        spec.distr='Gaussian';
        spec.dparams=0;
        spec.dtheta0=[];
    elseif ad==2
        spec.distr='T';
        spec.dparams=1;
        spec.dtheta0=10;
    else
        spec.distr='SkewT';
        spec.dparams=2;
        spec.dtheta0=[7;0];
    end
    spec.vecsize = 1+spec.ar+spec.ma + spec.vparams + spec.dparams;
    if aa~=3
        ae=1;%menu('define the PIT method','IFM','CML');
        if ae == 1
            spec.PIT = 'IFM';
        else
            spec.PIT = 'CML';
        end
    else
        spec.PIT = 'IFM';
    end
end




