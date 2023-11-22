%估计向量自回归模型的系数
function var_parameters = var_estimate(data , r)
[t , n] = size(data);
if r == 1
      temp1 = zeros(n,n+1);
      temp2 = zeros(n+1,n+1);
      for i = 2:t
          temp1 = temp1+data(i,:)'*[1 data(i-1,:)];
          temp2 = temp2+[1 data(i-1,:)]'*[1 data(i-1,:)];
      end
      var_parameters = temp1*((temp2)^(-1));
  end
  if r == 2
     temp1 = zeros(n,r*n+1);
     temp2 = zeros(r*n+1,r*n+1);
     for i = r+1:t
         temp1 = temp1+data(i,:)'*[1 data(i-1,:) data(i-2,:)];
         temp2 = temp2+[1 data(i-1,:) data(i-2,:)]'*[1 data(i-1,:) data(i-2,:)];
     end
     var_parameters = temp1*((temp2)^(-1));
 end
 if r == 3
     temp1 = zeros(n,r*n+1);
     temp2 = zeros(r*n+1,r*n+1);
     for i = r+1:t
         temp1 = temp1+data(i,:)'*[1 data(i-1,:) data(i-2,:) data(i-3,:)];
         temp2 = temp2+[1 data(i-1,:) data(i-2,:) data(i-3,:)]'*[1 data(i-1,:) data(i-2,:) data(i-3,:)];
     end
     var_parameters = temp1*((temp2)^(-1));
 end
if r > 3
    fprintf('r > 3 is difficult to estimate.');
end