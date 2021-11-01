# -*- coding: utf-8 -*-
# @Time :    2021/6/19  17:43
# @Author :  Eleven
# @Site :    
# @File :    data_final_normalization.py
# @Software: PyCharm
import pandas as pd

data = pd.read_csv('./Data/data_final.csv')

#归一化处理
data['Year'] = (data['Year']-data['Year'].mean())/data['Year'].std()
data['Floor'] = (data['Floor']-data['Floor'].mean())/data['Floor'].std()
data['Size'] = (data['Size']-data['Size'].mean())/data['Size'].std()
data.to_csv('./Data/data_final_normalization.csv',index=False)