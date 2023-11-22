function out = InputStartingValues(spec)


purp = spec.purpose;
switch purp
    case 'fitGARCH'
        defvals = [spec.mtheta0;spec.vtheta0;spec.dtheta0];
    case 'fitCopula'
        defvals = spec.ctheta0;
    case 'fitCopulaGARCH'
        n = spec.size;
        defvals = [repmat([spec.mtheta0;spec.vtheta0;spec.dtheta0],[n,1]);spec.ctheta0];
    case 'fitCopVine'
        defvals = spec.ctheta0;
end
out=defvals;
% aa=menu('define starting values','input defaults','keyboard input','already in workspace');
%    if aa==1
%         out=defvals;
%     elseif aa==2
%         out=input('type the starting values for the optimization:');
%     elseif aa==3
%         theta00=input('write the name of the variable in the workspace that contains the starting values:','s');
%         out=evalin('base',theta00);
%     end