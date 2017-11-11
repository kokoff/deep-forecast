import pandas as pd

data_ea = pd.read_excel('Data_ILP.xls', sheet_name=0, header=0, index_col=0, usecols=[i for i in range(3, 11)])
data_ea = data_ea[:][1:]
data_ea.columns = [i.upper() for i in data_ea.columns]
data_ea.to_csv('EA.csv')

data_us = pd.read_excel('Data_ILP.xls', sheet_name=1, header=0, index_col=0, usecols=[3, 4, 6, 8, 10, 9, 11, 12])
data_us = data_us[:][1:]
data_us = data_us[[u'CPI', u'GDP', u'UR', u'IR ', u'IR10', u'LR10 - IR', u' U.S. Dollars to One Euro']]
data_us.columns = data_ea.columns
data_us.to_csv('US.csv')
