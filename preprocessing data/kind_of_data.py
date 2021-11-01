# -*- coding: utf-8 -*-
# @Time :    2021/6/15  17:29
# @Author :  Eleven
# @Site :    
# @File :    kind_of_data.py
# @Software: PyCharm
import pandas as pd

data = pd.read_csv('data.csv',encoding='utf-8')
#获取数据种类
for _, column in data.iteritems():
    print(_)
    column = column.unique()
    print(column)
