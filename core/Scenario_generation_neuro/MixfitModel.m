function [parameters, LogL, evalmodel, GradHess, varargout] = MixfitModel(spec, data, solver)
% PURPOSE:  This function estimates with maximum likelihood the parameters
%           of the following models
%           1) univariate GARCH model(s)
%           2) Copulas
%           3) Copula - GARCH models in one step
%           4) Copula Vines
% To estimate a Copula GARCH model in two steps, first estimate the GARCH
% parameters and then the Copula parameters (define spec.purpose = 'fitGARCH'
% run fitmodel.m, then redefine spec.purpose = 'fitCopula' and run fitmodel.m
% again).
% INPUTS:
% spec:        Structure that contains model specifications, run
%              modelspec.m, to create it
% data:        [T,n] matrix of appropriate data
% solver:      String with values fmincon or fminunc. Fmincon is the
%              default however for standard errors better use fminunc
% OUTPUTS:
% parameters:   column vector of model parameters. In its full generality,
%               the parameters of each margin are put consecutively and
%               last are the copula parameters
% evalmodel:    Structure or cell that contains structures. It is the
%               output argument produced by fminunc, with Akaike and BIC
%               values for the corresponding model
% GradHess:     Structure or cell that contaiins structures that contains
%               the gradient and hessian at the optimum
% varargout:    When spec.purpose = 'fitGARCH' the standardized residuals
%               from the GARCH models are transformed to uniform. These
%               uniform variables are inputs to Copulas.
% ------------------------------------------------------------------------
% author: Manthos Vogiatzoglou, UoM
% contact at: vogia@yahoo.com
% ------------------------------------------------------------------------
%solver = 'fmincon';
if nargin == 2
    solver = 'fmincon';
end

switch solver
    case 'fmincon'
        % create starting values
        if isfield(spec,'comment')==1
            theta0 = spec.ctheta0;
        else
            theta0 = InputStartingValues(spec);
        end
        pause(.1);
        % create constraints + bounds matrices for fmincon
        [A, B, lower, upper] = CreateFminconConstraints(spec);
        %define optimization specifications
        options = optimset('Algorithm','interior-point','Display','iter','Hessian','bfgs','MaxFunEvals',12000);
        options = optimset(options,'FinDiffType','central','MaxIter',1500,'TolCon',10^-12,'TolFun',10^-5,'TolX',10^-5);
    case 'fminunc'
        %define optimization specifications
        options = optimset('Algorithm','interior-point','Display','iter','MaxFunEvals',9000,'MaxIter',1000,'TolCon',10^-12,'TolFun',10^-4,'TolX',10^-5,'FinDiffType','central');
        % invoke optimization procedure
end

purp = spec.purpose;
switch purp
    case 'fitGARCH'
        T = size(data,1); n = spec.size; m = spec.vecsize;
        parameters=zeros(m,n);
        LogL = zeros(1,n);
        exitflag = zeros(1,n);
        evalmodel = cell(1,n);
        GradHess = cell(1,n);
        udata = zeros(size(data));
        for i=1:n
            tic;
            if strcmp(solver,'fmincon')==1
                [parameters(:,i), LogL(i),exitflag(i),output,lambda,grad,hessian]= fmincon('GARCHLogL',theta0,A,B,[],[],lower,upper,[],options,data(:,i),spec,solver);
                %                 yz = menu('calculate asymptotic standard errors?','yes','no');
                %                 pause(.1)
                yz=0;
                if yz == 1
                    [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('GARCHLogL', parameters(:,i), data(:,i), grad,hessian,spec,solver);
                else
                    derivatives.grad = grad;
                    derivatives.hessian = hessian;
                end
            else
                % create starting values
                theta0 = InputStartingValues(spec);
                pause(.1);
                % make theta0 unconstrained
                theta0 = RescaleParameters(theta0, 2, spec);
                [params, LogL(i),exitflag(i),output,grad,hessian]= fminunc('GARCHLogL',theta0,options,data(:,i),spec,solver);
                parameters(:,i)=RescaleParameters(params,1,spec);
                %                 yz = menu('calculate asymptotic standard errors?','yes','no');
                %                 pause(.1)
                yz=0;
                if yz == 1
                    [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('GARCHLogL', parameters(:,i), data(:,i), grad, hessian, spec, solver);
                else
                    derivatives.grad = grad;
                    derivatives.hessian = hessian;
                    RobStE = [];
                end
            end
            if yz == 1
                derivatives.grad = grad;
                derivatives.hessian = hessian;
                GradHess{1,i}=derivatives;
            else
                GradHess{1,i}=derivatives;
            end
            RobStE=[];
            [AIC,BIC] = aicbic(-LogL(i),m,T);
            output.AIC = AIC;
            output.BIC = BIC;
            output.LogL = -LogL(i);
            evalmodel{1,i} = output;
            [dum, dum, dum, dum, udata(:,i)]=GARCHLogL(parameters(:,i),data(:,i),spec,'fmincon');
            output.TimeInSeconds = toc;
            DisplayResults(parameters(:,i),RobStE,output)
            %                 if i<n
            %                 fprintf(1,'Press any key to continue\n\n')
            %                 pause
            %                 end
            
        end
        varargout{1} = udata;
    case 'fitCopula'
        tic
        T = size(data,1); m = size(spec.ctheta0,1);
        if strcmp(solver,'fmincon')==1
            [parameters, LogL,exitflag,output,lambda,grad,hessian]= fmincon('MixCopulaLogL',theta0,A,B,[],[],lower,upper,[],options,data,spec,solver);
            [dum,dum,Rt]=MixCopulaLogL(parameters,data,spec, 'fmincon');
            if isfield(spec,'comment')==1
                yz = 0;
            else
                %yz = menu('calculate asymptotic standard errors?','yes','no');
                if strcmp(spec.family,'G-C')==1
                    yz=0;
                else
                    yz=0;
                end
            end
            %pause(.1)
            
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('MixCopulaLogL', parameters, data, grad, hessian, spec, 'fmincon');
            else
                derivatives.grad = grad;
                derivatives.hessian = hessian;
                RobStE = [];
            end
        else
            % create starting values
            theta0 = InputStartingValues(spec);
            pause(.1);
            % make theta0 unconstrained
            theta0 = RescaleParameters(theta0, 2, spec);
            [params, LogL,exitflag,output,grad,hessian]= fminunc('MixCopulaLogL',theta0,options,data,spec,solver);
            parameters=RescaleParameters(params,1,spec);
            [dum,dum,Rt]=MixCopulaLogL(parameters,data,spec, 'fmincon');
            %                 yz = menu('calculate asymptotic standard errors?','yes','no');
            %                 pause(.1)
            yz=0;
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('MixCopulaLogL', parameters, data, grad, hessian, spec, 'fminunc');
            else
                derivatives.grad = grad;
                derivatives.hessian = hessian;
                RobStE = [];
            end
        end
        if yz == 1
            derivatives.grad = grad;
            derivatives.hessian = hessian;
            GradHess=derivatives;
        else
            GradHess=derivatives;
        end
        GradHess.grad = grad;
        [AIC,BIC] = aicbic(-LogL,m,T);
        output.AIC = AIC;
        output.BIC = BIC;
        output.LogL = -LogL;
        output.Rt = Rt;
        output.TimeInSeconds = toc;
        evalmodel = output;
        DisplayResults(parameters,RobStE,output)
        varargout{1}=[];
    case 'fitCopulaGARCH'
        tic
        T = size(data,1); m = size(spec.ctheta0,1);
        if strcmp(solver,'fmincon')==1
            [parameters, LogL,exitflag,output,lambda,grad,hessian]= fmincon('MixCopulaGARCHLogL',theta0,A,B,[],[],lower,upper,[],options,data,spec,solver);
            [dum, dum, tvpars] = MixCopulaGARCHLogL(parameters,data,spec,'fmincon');
            %                yz = menu('calculate asymptotic standard errors?','yes','no');
            %                pause(.1)
            yz=0;
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('MixCopulaGARCHLogL', parameters, data, grad, hessian, spec, 'fmincon');
            end
        else
            % create starting values
            theta0 = InputStartingValues(spec);
            %theta0 = spec.theta0;
            pause(.1);
            % make theta0 unconstrained
            theta0 = RescaleParameters(theta0, 2, spec);
            [params, LogL,exitflag,output,grad,hessian]= fminunc('MixCopulaGARCHLogL',theta0,options,data,spec,solver);
            parameters=RescaleParameters(params,1,spec);
            [dum, dum, tvpars] = MixCopulaGARCHLogL(parameters,data,spec,'fmincon');
            %yz = menu('calculate asymptotic standard errors?','yes','no');
            yz = 1;
            %pause(.1)
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('MixCopulaGARCHLogL', parameters, data, grad, hessian,spec, 'fminunc');
            else
                derivatives.grad = grad;
                derivatives.hessian = hessian;
            end
        end
        if yz == 1
            derivatives.grad = grad;
            derivatives.hessian = hessian;
            GradHess=derivatives;
        else
            GradHess=derivatives;
        end
        GradHess.grad = grad;
        [AIC,BIC] = aicbic(-LogL,m,T);
        output.AIC = AIC;
        output.BIC = BIC;
        output.LogL = -LogL;
        output.TimeInSeconds = toc;
        output.exitflag = exitflag;
        output.tvpars = tvpars;
        evalmodel = output;
        DisplayResults(parameters,RobStE,output)
        varargout{1}=[];
    case 'fitCopVine'
        tic
        T = size(data,1); m = size(spec.ctheta0,1);
        if strcmp(solver,'fmincon')==1
            [parameters, LogL,exitflag,output,lambda,grad,hessian]= fmincon('CopulaVineLogL',theta0,A,B,[],[],lower,upper,[],options,data,spec,solver);
            %                yz = menu('calculate asymptotic standard errors?','yes','no');
            %                pause(.1)
            yz=0;
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('CopulaVineLogL', parameters, data, grad, hessian, spec, 'fmincon');
            end
        else
            % create starting values
            theta0 = InputStartingValues(spec);
            pause(.1);
            % make theta0 unconstrained
            theta0 = RescaleParameters(theta0, 2, spec);
            [params, LogL,exitflag,output,grad,hessian]= fminunc('CopulaVineLogL',theta0,options,data,spec,solver);
            parameters=RescaleParameters(params,1,spec);
            %                 yz = menu('calculate asymptotic standard errors?','yes','no');
            %                 pause(.1)
            yz=0;
            if yz == 1
                [derivatives, RobVCV, VCV, hessian, RobStE]=CalcStErrors('CopulaVineLogL', parameters, data, grad, hessian,spec, 'fminunc');
            else
                derivatives.grad = grad;
                derivatives.hessian = hessian;
                RobStE=[];
            end
        end
        if yz == 1
            derivatives.grad = grad;
            derivatives.hessian = hessian;
            GradHess=derivatives;
        else
            GradHess=derivatives;
        end
        GradHess.grad = grad;
        [AIC,BIC] = aicbic(-LogL,m,T);
        output.AIC = AIC;
        output.BIC = BIC;
        output.LogL = -LogL;
        output.TimeInSeconds = toc;
        evalmodel = output;
        DisplayResults(parameters,RobStE,output)
        varargout{1}=[];
end
%end
