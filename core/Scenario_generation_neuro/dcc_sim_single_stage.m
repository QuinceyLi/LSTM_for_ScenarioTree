function [finalu,uu] = dcc_sim_single_stage(k,t,u,stock,Qbar,hmat,Qt,garchParameters,garchP,garchQ,dccparameters,dccP,dccQ,r,bb)
% k 为股票数目
% u 样本内残差
% t 模拟阶段数

%%%%%%%%%%
dccA = dccparameters(1:dccP);
dccB = dccparameters(dccP+1:dccQ+dccP);
sumA = sum(dccA);
sumB = sum(dccB);

m = size(stock,1)-t+1;
len = size(u,1)-t+1;

for ii = 1:t
    index = 1;
    for j = 1 : k
        para = garchParameters( index:index + garchP( j ) + garchQ( j ) );
        index = index + 1 + garchP( j ) + garchQ( j );
        
        if garchP(j) == 1 & garchQ(j) == 1
            hh( 1 , j ) = para' * [ 1 ; u( len+ii-1 , j ).^2 ; hmat( len+ii-1 , j ) ];
        elseif garchP(j) == 1 & garchQ(j) == 2
            hh( 1 , j ) = para' * [ 1 ; u( len+ii-1 , j ).^2 ; hmat( len+ii-1 , j ) ; hmat( len+ii-2 , j )];
        elseif garchP(j) == 2 & garchQ(j) == 1
            hh( 1 , j ) = para' * [ 1 ;u( len+ii-1 , j ).^2 ; u( len+ii-2 , j ).^2 ; hmat( len+ii-1 , j ) ];
        elseif garchP(j) == 2 & garchQ(j) == 2
            hh( 1 , j ) = para' * [ 1 ;u( len+ii-1 , j ).^2 ;  u( len+ii-2 , j ).^2 ; hmat( len+ii-1 , j ) ; hmat( len+ii-2 , j ) ];
        else
            print('p>2 and q>2 are difficulty to estimate!\n');
        end
    end
    
    Dt = diag(sqrt(hh));
    hmat(len+ii , :) = hh;     %方差
    
    for i = 1:dccP
        stdresid(i ,:) = u(len-i+ii ,:)./sqrt(hmat(len-i+ii,:));
    end
    
    % cc = garchParameters(index:index+dccP-1);
    % dd = garchParameters(index+dccP:index+dccP+dccQ-1);
    % sumA = sum(cc);
    % sumB = sum(dd);
    
    Qt(:,:,len+ii) = Qbar*(1-sumA-sumB);
    for i=1:dccP
        Qt(:,:,len+ii) = Qt(:,:,len+ii)+dccA(i)*(stdresid( i,:)'*stdresid( i,:));
    end
    for i = 1:dccQ
        Qt(:,:,len+ii) = Qt(:,:,len+ii)+dccB(i)*Qt(:,:,len+ii-i);
    end
    Rt = Qt(:,:,len+ii)./(sqrt(diag(Qt(:,:,len+ii)))*sqrt(diag(Qt(:,:,len+ii)))');
    
    Ht = Dt*Rt*Dt;  %方差-协方差矩阵
    if ii==t
        uu = zeros( k , 1 );
        
        uu = mvnrnd(uu,Ht,1)';
        
        % u(len+ii,:) = uu';
        
        if r == 1
            temp = bb*[1,stock(m+ii-1 ,:)]'+uu;
            finalu = temp';
        end
        if r == 2
            temp = bb*[1 stock(m+ii-1 ,:) stock(m+ii-2 ,:)]'+uu;
            finalu = temp';
        end
        if r == 3
            temp = bb*[1,stock(m+ii-1 , :) stock(m+ii-2 , :) stock(m+ii-3 , :)]'+uu;
            finalu = temp';
        end
    end
    %u(len+ii,:) = stock(m+ii , :);
end