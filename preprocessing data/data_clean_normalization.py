# -*- coding: utf-8 -*-
# @Time :    2021/6/22  23:38
# @Author :  Eleven
# @Site :    
# @File :    data_clean_normalization.py
# @Software: PyCharm
import pandas as pd
import numpy as np


data = pd.read_csv('./Data/data_clean.csv')

#归一化处理
data['Year'] = (data['Year']-data['Year'].mean())/data['Year'].std()
data['Floor'] = (data['Floor']-data['Floor'].mean())/data['Floor'].std()
data['Size'] = (data['Size']-data['Size'].mean())/data['Size'].std()
data['Price'] = (data['Price']-data['Price'].mean())/data['Price'].std()
data.to_csv('./Data/data_clean_normalization.csv')