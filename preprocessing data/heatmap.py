# -*- coding: utf-8 -*-
# @Time :    2021/6/17  17:41
# @Author :  Eleven
# @Site :    
# @File :    heatmap.py
# @Software: PyCharm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/data_clean.csv')

#汉字和负号输出
sns.set(font='simhei')
plt.rcParams['axes.unicode_minus'] = False

def heatmap_origin():
    plt.figure(figsize=(36,27))
    corr = data.corr()
    sns.heatmap(corr[(corr>=0.3)|(corr<=-0.3)],cmap='Greys',annot=True,linewidths=1,linecolor='b')
    plt.title('相关性系数矩阵热力图',fontsize=30)
    plt.savefig('./img/heatmap_origin.jpg')

def heatmap():
    data = pd.read_csv('./Data/data_without_onehot.csv')
    plt.figure(figsize=(12,9))
    corr = data.corr()
    sns.heatmap(corr,cmap='Greys',annot=True)
    plt.title('相关性系数矩阵热力图',fontsize=30)
    plt.savefig('./img/heatmap.jpg')

if __name__ == '__main__':
    heatmap_origin()
    heatmap()


