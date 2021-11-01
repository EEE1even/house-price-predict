# -*- coding: utf-8 -*-
# @Time :    2021/6/22  22:35
# @Author :  Eleven
# @Site :    
# @File :    data_final.py
# @Software: PyCharm
import pandas as pd

data_not_null = pd.read_csv('./Data/data_knn.csv')
data = pd.read_csv('./Data/data_clean.csv')

#将knn算法填充后的数据与data_clean合并
data_not_null = data_not_null.loc[:,['Elevator','Hall','Id']]
data = data.drop(['Elevator','Hall'],axis=1)
data = pd.merge(data,data_not_null,how='left',on='Id')
data.to_csv('./Data/data_final.csv',index=False)