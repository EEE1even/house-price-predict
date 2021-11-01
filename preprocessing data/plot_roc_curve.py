# -*- coding: utf-8 -*-
# @Time :    2021/6/25  3:03
# @Author :  Eleven
# @Site :    
# @File :    plot_roc_curve.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


'''
roc曲线绘制
对比跑分
'''
fig,ax = plt.subplots()
paths = [("sta","./Data/data_clean.csv"),("nor","./Data/data_clean_normalization.csv")]
for name,path in paths:
    data = pd.read_csv(path)
    data = data[data['Elevator'].notnull()].loc[:,['Floor','Price','Size','Year','Room','Elevator']]
    train = data.sample(frac=0.8,random_state=0,axis=0)
    dev = data[~data.index.isin(train.index)]
    x_train = train.iloc[:,:-1]
    y_train = train['Elevator']
    x_dev = dev.iloc[:,:-1]
    y_dev = dev['Elevator']
    clf = KNeighborsClassifier(n_neighbors=5,weights="distance")
    clf.fit(x_train,y_train)
    metrics.plot_roc_curve(clf, x_dev, y_dev,ax=ax,name=name)
plt.savefig('./img/plot_roc_curve.jpg')
plt.show()

#
