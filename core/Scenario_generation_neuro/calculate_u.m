function now_u=calculate_u(now_data,stock,bb)

% nowdata�ֽ׶ε�����������
%stockΪ��ʷ����������
%%%%%%%%%%

now_u = now_data - (bb*[1 stock(end,:)]')';

    