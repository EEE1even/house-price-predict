# -*- coding: utf-8 -*-
# @Time :    2021/6/24  0:02
# @Author :  Eleven
# @Site :    
# @File :    data_assessment.py
# @Software: PyCharm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

data = pd.read_csv('./Data/data_clean.csv')
sns.set(font='simhei')
plt.rcParams['axes.unicode_minus'] = False

def Price_pairplot():
    for i in tqdm(range(10)):
        time.sleep(0.1)
        sns.set(font='simhei')
        plt.rcParams['axes.unicode_minus'] = False
        sns.pairplot(data, x_vars=data.columns.values[10*i:10*(i+1)], y_vars="Price")
        plt.savefig('./img/pairplot{}.jpg'.format(i))

def Price_Year_boxplot():
    data_train = pd.concat([data['Price'],data['Year']],axis=1)
    plt.figure(figsize=(12,9))
    sns.boxplot(x='Year',y='Price',data=data_train)
    plt.axis(ymin=0,ymax=7000)
    plt.xticks(rotation=90)
    plt.savefig('./img/Price_Year_boxplot.jpg')
    plt.show()


def nan_view():

    data_null = (data.isnull().sum()/len(data))*100

    data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:30]

    missing_data = pd.DataFrame({'Missing Data':data_null})

    m = missing_data.head(5)
    print(m)
    plt.figure(figsize=(12,16))
    plt.xticks(rotation = 90,fontsize=20)
    plt.yticks(fontsize=20)
    sns.barplot(x=data_null.index,y=data_null)
    plt.xlabel('缺失值列名',fontsize=25)
    plt.ylabel('占总数据的百分比(%)',fontsize=25)
    plt.title('缺失值占比柱状图',fontsize=42)
    plt.savefig('./img/nan_view.jpg')
    plt.show()

#处理后的图很脏，看不了
def Price_boxplot():
    categorical = data.select_dtypes(exclude=['int64','float64'])
    def facetgrid_boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x=plt.xticks(rotation=90)

    f = pd.melt(data, id_vars=['Price'], value_vars=sorted(data[categorical.columns]))
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5)
    g.map(facetgrid_boxplot, "value", "Price")
    plt.show()

if __name__ == '__main__':
    # Price_Year_boxplot()
    # Price_pairplot()
    nan_view()