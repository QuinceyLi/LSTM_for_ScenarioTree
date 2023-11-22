clc
clear
%%%%%%%%%%%%%%%%%%%%% AMZN  5935
A = readmatrix('AMZN_20000101.csv', 'OutputType', 'string');
data1=A(:,2:end);
[row,col]=size(data1);
AMZN_data=zeros(row,col);
for i=1:row
    for j=1:col
        AMZNE_data(i,j)=str2num(data1(i,j));
    end
end
%%%%%%%%%%%%%%%%%%%%% EA  5935
A = readmatrix('EA20000101.csv', 'OutputType', 'string');
data1=A(:,2:end);
[row,col]=size(data1);
EA_data=zeros(row,col);
for i=1:row
    for j=1:col
        EA_data(i,j)=str2num(data1(i,j));
    end
end
%%%%%%%%%%%%%%%%%%%%% NTDOY  5935
A = readmatrix('NTDOY20000101.csv', 'OutputType', 'string');
data1=A(:,2:end);
[row,col]=size(data1);
NTDOY_data=zeros(row,col);
for i=1:row
    for j=1:col
        NTDOY_data(i,j)=str2num(data1(i,j));
    end
end
%%%%%%%%%%%%%%%%%%%%% SONY  5935
A = readmatrix('SONY20000101.csv', 'OutputType', 'string');
data1=A(:,2:end);
[row,col]=size(data1);
SONY_data=zeros(row,col);
for i=1:row
    for j=1:col
        SONY_data(i,j)=str2num(data1(i,j));
    end
end
%%%%%%%%%%%%%%%%%%%%% TTWO  5935
A = readmatrix('TTWO20000101.csv', 'OutputType', 'string');
data1=A(:,2:end);
[row,col]=size(data1);
TTWO_data=zeros(row,col);
for i=1:row
    for j=1:col
        TTWO_data(i,j)=str2num(data1(i,j));
    end
end
%%%%%%%%%%%%%%%%% CLOSE DATA
close_data=zeros(row,5);n_data=
close_data(:,1)=AMZNE_data(:,4);
close_data(:,2)=EA_data(:,4);
close_data(:,3)=NTDOY_data(:,4);
close_data(:,4)=SONY_data(:,4);
close_data(:,5)=TTWO_data(:,4);
return_data=(close_data(2:end,:)-close_data(1:end-1,:))./close_data(1:end-1,:);
save stock_data  return_data close_data
