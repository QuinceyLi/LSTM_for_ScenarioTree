import akshare as ak
from scipy.io import savemat
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
# sns.set_theme(style='darkgrid') # 设置风格使图标更美观
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
def get_stock():
    S = []
    for i in ['01','12','14','21','25']:
        symbol = "0000" + i
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="19981001", end_date='20231130', adjust="")
        stock_return = stock_zh_a_hist_df[['收盘']].pct_change()[1:5935]
        S.append(stock_return['收盘'].values)

    # print(S)
    # savemat('china_stock.mat', {'return_data':np.array(S).T})

    np.savetxt('cn_return_data.csv',np.array(S).T,delimiter=',')

def plot_stock(filename):
    df = pd.read_csv(filename)
    # stock = df.get('stock1','stock2','stock3','stock4','stock5').values()
    df = df.iloc[-238:]
    df['time'] = pd.date_range(start='2001-01-01', periods=238, freq='D')
    df.set_index('time', inplace=True)
    df[['stock1']].plot(figsize=(12,8))
    plt.show()

if __name__ == '__main__':
    plot_stock('../data/cn_return_data.csv')