function DisplayResults(params,RobStE,output)

if isempty(RobStE) == 1
    tstats = [];
    flag = 1;
else
    n = size(params,1);
    if size(RobStE,1)~=n
    error('params and RobStE should have the same size')
    end
    tstats = params./RobStE;
    flag = 2;
end

fprintf(1,'\n Estimation output \n')
fprintf(1, 'parameter   St. Error    t-stats\n')
fprintf(1,'---------------------------------\n')
if flag == 2
    fprintf(1,'%3.4f\t\t %2.3f\t\t %3.4f\t\t\t\n', [params, RobStE, tstats]' )
else
    fprintf(1,'%3.4f\t\t \n', params')
end
fprintf(1,'---------------------------------\n')
fprintf(1,'Akaike: %5.4f \n',output.AIC)
fprintf(1,'BIC: %5.4f \n',output.BIC)
fprintf(1,'Log Likelihood: %5.3f\n',output.LogL)
fprintf(1,'---------------------------------\n')
fprintf(1,'Estimation time is %4.2f seconds\n',output.TimeInSeconds)
