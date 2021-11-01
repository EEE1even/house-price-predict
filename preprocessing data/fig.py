# -*- coding: utf-8 -*-
# @Time :    2021/6/23  14:44
# @Author :  Eleven
# @Site :    
# @File :    fig.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('./Data/data_without_onehot.csv')
plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def Elevator_Floor_scatter():
    """
        电梯与楼层的关系散点图
    """
    data_have_elevator = np.array(data[(data['Elevator']==1.0)]['Floor'].values)
    data_no_elevator = np.array(data[(data['Elevator']==0.0)]['Floor'].values)

    Elevator_have = np.arange(1,len(data_have_elevator)+1,dtype=int)
    Elevator_no = np.arange(1,len(data_no_elevator)+1,dtype=int)
    plt.figure(figsize=(15,9))
    plt.title('有无电梯与楼层高度的关系散点图',fontsize=40)
    #绘制散点图
    plt.scatter(Elevator_have,data_have_elevator,alpha=0.7)
    plt.scatter(Elevator_no,data_no_elevator,c='r',alpha=1,marker='^')
    #设置坐标轴字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=21)
    #设置横纵坐标标签
    plt.xlabel('户主个数',fontsize=25)
    plt.ylabel('楼层数',fontsize=25)
    #添加标签
    plt.legend(['有电梯','无电梯'],fontsize=18)
    plt.savefig('./img/Elevator_Floor_scatter.jpg')
    plt.show()

def Price_Size_scatter():
    """
    价格和大小的关系散点图
    :return:
    """
    plt.figure(figsize=(12,9))
    sns.scatterplot(x=data['Price'], y=data['Size'], hue='Size',data=data,palette='coolwarm')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('房屋大小与价格的关系散点图',fontsize=40)
    plt.xlabel('价格（万元）',fontsize = 22)
    plt.ylabel('大小（平方米）',fontsize = 22)
    plt.savefig('./img/Price_Size_scatter.jpg')
    plt.show()

def Price_Year_plot():
    """
    价格和年份的关系折线图
    :return:
    """
    data_year = data['Year'].unique()
    #创建列表
    mean_price = []
    #sort从小到大排序
    data_year.sort()

    #整体数据按照年份升序排序
    # data_sort_year = data.sort_values(by='Year',ascending=True)
    # #ascending=True升序 False降序

    #for循环求各个年份的房价平均值
    for i in data_year:
        temp =  data[data['Year']==i]['Price']
        mean = temp.mean()#求平均
        mean_price.append(mean)

    plt.figure(figsize=(12,9))
    plt.title('历年房价平均值折线图',fontsize=40)
    plt.xlabel('年份',fontsize=25)
    plt.ylabel('价格（万元）',fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(data_year,mean_price,c='deeppink',linewidth=3)
    plt.savefig('./img/Price_Year_plot.jpg')
    plt.show()

def Elevator_Year_scatter():
    data_have_elevator = np.array(data[(data['Elevator']==1.0)]['Year'].values)
    data_no_elevator = np.array(data[(data['Elevator']==0.0)]['Year'].values)

    Elevator_have = np.arange(1,len(data_have_elevator)+1,dtype=int)
    Elevator_no = np.arange(1,len(data_no_elevator)+1,dtype=int)
    plt.figure(figsize=(15,9))
    plt.title('有无电梯与年份的关系散点图',fontsize=40)
    #绘制散点图
    plt.scatter(Elevator_have,data_have_elevator,alpha=0.7)
    plt.scatter(Elevator_no,data_no_elevator,c='r',alpha=1,marker='^')
    #设置坐标轴字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=21)
    #设置横纵坐标标签
    plt.xlabel('户主个数',fontsize=25)
    plt.ylabel('年份',fontsize=25)
    #添加标签
    plt.legend(['有电梯','无电梯'],fontsize=18)
    plt.savefig('./img/Elevator_Year_scatter.jpg')
    plt.show()

if __name__ == '__main__':
    Elevator_Year_scatter()
    Elevator_Floor_scatter()
    Price_Size_scatter()
    Price_Year_plot()
