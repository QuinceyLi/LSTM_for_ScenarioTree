# LSTM_for_ScenarioTree
**Data**:5类股票收益率数据

**Prior Assumption**:股票每日的收益率满足某一个先验分布(Gaussian Distribution)

**LSTM**:利用LSTM预测先验分布的方差和均值，抽样生成情景树

**Application**:利用生成的情景树做投资决策，通过回测曲线判断决策的好坏
