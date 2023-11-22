function now_u=calculate_u(now_data,stock,bb)

% nowdata现阶段的收益率数据
%stock为历史收益率数据
%%%%%%%%%%

now_u = now_data - (bb*[1 stock(end,:)]')';

    