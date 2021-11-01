# -*- coding: utf-8 -*-
# @Time :    2021/6/8  16:53
# @Author :  Eleven
# @Site :    
# @File :    analysis.py
# @Software: PyCharm

import pandas as pd


data = pd.read_csv('data.csv',encoding='utf-8')
#替换电梯的参数
data = data.replace('有电梯',1)
data = data.replace('无电梯',0)

#替换装修的参数
data = data.replace('精装',3)
data = data.replace('简装',2)
data = data.replace('毛坯',1)
data = data.replace('其他',0)

#房屋朝向数据onehot处理
data_Direction = pd.get_dummies(data['Direction'])
data = data.join(data_Direction)

'''
    所在地区用onehot处理,并合并表格
'''
data_region = pd.get_dummies(data['Region'])
data = data.join(data_region)

'''
    删除不必要数据 处理Renovation列数据(装修)
    '精装' =  2
    '简装' =  1
    '毛坯' =  0
    '其他' = -1
'''
data = data[data['Renovation']!='南北']
data_renovation = pd.get_dummies(data['Renovation'])
data = data.join(data_renovation)
data = data.rename(columns={3:'精装',2:'简装',1:'毛坯',0:'其他'})


'''
    拆分Layout(布局)为单独列，按照室(Room)，厅(Hall)划分
'''
#房间卫生间拆分
m = data[data['Layout'].str.contains('房间')]
m = m.loc[:,['Id','Layout']]#保留想要的列
#拆分
m.loc[:,'Layout'] = m['Layout'].str.replace('房间','|').astype('str')
m.loc[:,'Layout'] = m['Layout'].str.replace('卫','').astype('str')
x = m['Layout'].str.split('|',expand=True)#expand=True分割完不是[]格式
m = m.join(x)
m = m.drop(['Layout'],axis=1)
m = m.rename(columns={0:'Room',1:'Toilet'})
#室厅拆分
n = data[data['Layout'].str.contains('厅')]
n = n.loc[:,['Id','Layout']]
n.loc[:,'Layout'] = n['Layout'].str.replace('室','|').astype('str')
n.loc[:,'Layout'] = n['Layout'].str.replace('厅','').astype('str')
y = n['Layout'].str.split('|',expand=True)#expand=True分割完不是[]格式
n = n.join(y)
n = n.drop(['Layout'],axis=1)
n = n.rename(columns={0:'Room',1:'Hall'})


'''
将拆分后的两列合并，删除toilte列
以Id为合并依据，按照left形式合并
'''
data = pd.merge(data,m,how='left',on='Id')
data = pd.merge(data,n,how='left',on='Id')
data = data.drop(['Toilet'],axis=1)

'''
由于上面拆分出的数据分为 “ 室 厅” 数据和 “ 房间 卫生间” 数据
整合后有互相交错的nan集，所以将[房间]列中的数据填充到[室]列的nan中
删除[卫生间]列
'''
for i in range(len(data['Room_x'])):
    if  pd.isna(data['Room_x'][i]):
        data.loc[i,'Room_x'] = data.loc[i,'Room_y']#需要用loc提取数据，否则会报错
data = data.drop(['Room_y'],axis =1)
data = data.rename(columns={'Room_x':'Room'})#修改列名

#保存
data.to_csv('./Data/data_clean.csv',index=False)#index为False去除索引