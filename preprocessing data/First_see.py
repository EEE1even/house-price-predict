# -*- coding: utf-8 -*-
# @Time :    2021/6/30  16:52
# @Author :  Eleven
# @Site :    
# @File :    First_see.py
# @Software: PyCharm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def one():
    df = pd.read_csv("data.csv")
    #根据数据类型筛选，只选择int64和float64
    data_numbers = df.select_dtypes(include=['int64','float64'])
    #选择object对象存入df
    df = df.select_dtypes(include=['object'])

    #创建一个空的DATa Frame 里面有columns[]和Index[]
    data = pd.DataFrame()


    for i in range(len(df.columns)):
        #将离散型的数据转换成 0到 n−1 之间的数，这里 n是一个列表的不同取值的个数，可以认为是某个特征的所有不同取值的个数。
        #硬编码
        l = LabelEncoder()
        l.fit(df.iloc[:,i])
        label = l.transform(df.iloc[:,i])
        label = pd.Series(label,name=df.columns[i])
        data = pd.concat([data,label],axis=1)

    data =pd.concat([data,data_numbers],axis=1)

    fig,axs= plt.subplots(4,3,figsize=(20,25),dpi=144)

    #调整纵向间距和平均高度的比值
    fig.subplots_adjust(hspace=0.6)

    for i,ax in zip(data.columns,axs.flatten()):
        #在用plt.subplots画多个子图中，ax = ax.flatten()将ax由n*m的Axes组展平成1*nm的Axes组
        sns.scatterplot(x=i, y='Price', hue='Price',data=data,ax=ax,palette='coolwarm')
        plt.xlabel(i,fontsize=12)
        plt.ylabel('Price',fontsize=12)
        ax.set_title('Price'+' - '+str(i),fontweight='bold',size=20)
    plt.savefig('./img/First_see_scatter.jpg')
    plt.show()

def two():
    df = pd.read_csv("data.csv")
    data_numbers = df.select_dtypes(include=['int64','float64'])
    data_object = df.select_dtypes(include=['object'])
    data = pd.DataFrame()
    for i in range(len(data_object.columns)):
        l = LabelEncoder()
        l.fit(data_object.iloc[:,i])
        label = l.transform(data_object.iloc[:,i])
        label = pd.Series(label,name=data_object.columns[i])
        data = pd.concat([data,label],axis=1)
    data = pd.concat([data,data_numbers],axis=1)
    def facetgrid_boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x=plt.xticks(rotation=90)


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    # print(sorted(df[data.columns]))
    f = pd.melt(df, id_vars=['Price'], value_vars=['Direction', 'Elevator', 'Floor', 'Layout', 'Region', 'Renovation', 'Year'])
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5,)
    g = g.map(facetgrid_boxplot, "value", "Price")
    plt.savefig('./img/First_see_box.jpg')
    plt.show()

if __name__ == '__main__':
    two()