# -*- coding: utf-8 -*-
# @Time :    2021/6/22  10:30
# @Author :  Eleven
# @Site :    
# @File :    data_knn.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

def knn_missing_filled(x_train,y_train,test,k=5,dispersed=True):
    """
    定义knn填充空值函数
    x_train,y_train为训练集，test为需要填充的数据
    dispersed=True时，填充值为整数
    dispersed=False时，填充值为精确值
    """
    if dispersed:
        '''
        weight为权重,有distance和unifrom;
        distance
        权重点与距离的倒数。在这种情况下，查询点的近邻将比远的近邻具有更大的影响。
        unifrom
        所有分数都是平等的
        '''
        clf = KNeighborsClassifier(n_neighbors=k,weights="distance")
        #weight为权重,有distance和unifrom;distance
        """
            交叉验证
            :param为最优化的参数取值
            cv 指定flod数量
        """
        param={
            "n_neighbors":[1,3,5],
        }
        cv = GridSearchCV(clf,param_grid=param,cv=2)
        cv.fit(x_train,y_train)
        print(cv.best_score_)
    else:
        clf = KNeighborsRegressor(n_neighbors=k,weights="distance")

    clf.fit(x_train,y_train)

    return test.index,clf.predict(test)

#保存所需要的列
data = pd.read_csv('./Data/data_clean.csv')
data = data.loc[:,['Elevator','Floor','Price','Size','Year','Room','Hall','Id']]

#删除其他列（排除要填充的列）中存在空值的行,并保存删除的数据
#如果没有空值，则忽略该步骤的结果
del_Data = data[(data['Floor'].isnull())|(data['Price'].isnull())|(data['Size'].isnull())|(data['Year'].isnull())|(data['Room'].isnull())]
del_Data.to_csv('./Data/knn_del_data.csv',index=False)


#[电梯]列nan填充
Elevator_x_train = data[data['Elevator'].notnull()].loc[:,['Floor','Price','Size','Year','Room']]
Elevator_y_train = data[data['Elevator'].notnull()]['Elevator']
Elevator_test = data[data['Elevator'].isnull()].loc[:,['Floor','Price','Size','Year','Room']]
Elevator_index,Elevator_value = knn_missing_filled(Elevator_x_train,Elevator_y_train,Elevator_test,k=5,dispersed=True)
data.loc[Elevator_index,'Elevator'] = Elevator_value

#[厅]列nan填充
Hall_x_train = data[data['Hall'].notnull()].loc[:,['Floor','Price','Size','Year','Room']]
Hall_y_train = data[data['Hall'].notnull()]['Hall']
Hall_test = data[data['Hall'].isnull()].loc[:,['Floor','Price','Size','Year','Room']]
Hall_index,Hall_value = knn_missing_filled(Hall_x_train,Hall_y_train,Hall_test,k=5,dispersed=True)
data.loc[Hall_index,'Hall'] = Hall_value

data_clean = pd.read_csv('./Data/data_clean.csv')
data['Year'] = data['Year']*data_clean['Year'].std()+data_clean['Year'].mean()
data['Floor'] = data['Floor']*data_clean['Floor'].std()+data_clean['Floor'].mean()
data['Size'] = data['Size']*data_clean['Size'].std()+data_clean['Size'].mean()
data['Price'] = data['Price']*data_clean['Price'].std()+data_clean['Price'].mean()
data = data.round(1)#取小数点后一位

#
data.to_csv('./Data/data_knn.csv',index=False)
