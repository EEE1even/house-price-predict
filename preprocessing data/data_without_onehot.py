# -*- coding: utf-8 -*-
# @Time :    2021/6/22  23:51
# @Author :  Eleven
# @Site :    
# @File :    data_without_onehot.py
# @Software: PyCharm
import pandas as pd


data = pd.read_csv('./Data/data_final.csv')
data = data.loc[:,['Floor','Price','Renovation','Year','Size','Elevator','Room','Hall','Id']]

data.to_csv('./Data/data_without_onehot.csv',index=False)