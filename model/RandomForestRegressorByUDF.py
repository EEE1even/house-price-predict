# _*_ coding utf-8 _*_
# @Time     : 2021/6/28 17:05
# @Author   : yaocctao
import time

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd


class HousePricePredictByRandomForestRegressor():
    def __init__(self,n_estimators=500,max_depth=30,oob_score=True):
        self.clf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, random_state=0,oob_score=oob_score)

    def get_data(self,data,frac=0.8):
        data = data[['Size', 'Year', 'Room','Direction','District','Garden', 'Region', 'Floor',
                     'Renovation', 'Elevator', 'Hall','Price']]
        trainData = data.sample(frac=frac, random_state=0, axis=0)
        x_train = trainData.iloc[:, :-1]
        y_train = trainData["Price"]
        devData = data[~data.index.isin(trainData.index)]
        x_dev = devData.iloc[:, :-1]
        y_dev = devData["Price"]

        return x_train,y_train,x_dev,y_dev,trainData,devData,data

    def train(self,x_train,y_train):
        return self.clf.fit(x_train,y_train)

    def dev(self,x_dev,y_dev):
        return self.clf.score(x_dev,y_dev)

    def predict(self,x_data):
        return self.clf.predict(x_data)

    def plot_fit_curve(self,x_data,y_data,y_pred,i=0):
        plt.figure(figsize=(16, 10), dpi = 144)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.scatter(x_data.iloc[:, i], y_data, c='b', label='data',alpha=0.8) #训练样本
        plt.scatter(x_data.iloc[:, i], y_pred, label='prediction',c="darkorange", lw=2,alpha=0.8) #拟合曲线
        plt.axis('tight')
        plt.title('RandomForestRegressor regression (k =%f)' % self.clf.score(x_data,y_data))
        plt.xlabel(x_data.columns[i])
        plt.ylabel("Prices")
        plt.show()

    def heighten_data(self,data,frac=0.8):
        data = data[['Size', 'Year', 'Room','Direction','District','Garden', 'Region', 'Floor',
                     'Renovation', 'Elevator', 'Hall','Price']]


        #         data = pd.concat([data,extra_data],axis=0)
        trainData = data.sample(frac=frac, random_state=0, axis=0)
        devData = data[~data.index.isin(trainData.index)]
        extra_data = 0.5*trainData[trainData["Price"]>2000].copy()
        # print([extra_data.index[0]])
        # print(pd.DataFrame(extra_data.loc[extra_data.index[0:2]].mean(),columns=['Size', 'Year', 'Room','Direction','District','Garden', 'Region', 'Floor',
        #                                                                                    'Renovation', 'Elevator', 'Hall','Price']))
        # print(extra_data.loc[extra_data.index[1]].add(extra_data.loc[extra_data.index[0]]).div(2))
        new_data = pd.DataFrame(columns=['Size', 'Year', 'Room','Direction','District','Garden', 'Region', 'Floor',
                                         'Renovation', 'Elevator', 'Hall','Price'])
        length = len(extra_data)
        for i in range(length):
            if i < length-1:
                new_data = pd.concat([new_data,extra_data.loc[extra_data.index[i:i+2]].mean().to_frame().unstack().unstack()],axis=0)
            else:
                break
        new_data.index=range(len(data),len(data)+len(new_data))
        trainData = pd.concat([trainData,new_data],axis=0)

        x_train = trainData.iloc[:, :-1]
        y_train = trainData["Price"]

        x_dev = devData.iloc[:, :-1]
        y_dev = devData["Price"]

        return x_train,y_train,x_dev,y_dev,trainData,devData,data


    def Cross_Validation(self,data,param):
        start = time.time()
        clf = RandomForestRegressor(oob_score=True)
        cv = GridSearchCV(clf,param_grid=param,cv=5)
        cv.fit(data.iloc[:, :-1],data["Price"])
        print(cv.best_score_)
        print(cv.best_params_)
        end = time.time()
        print("time:"+str(start-end))
        return cv.best_estimator_

if __name__ == '__main__':

    #实例化模型
    clf = HousePricePredictByRandomForestRegressor()
    #读取数据
    data =pd.read_csv("./data/encode_data_addMissHall.csv")
    # x_train,y_train,x_dev,y_dev,trainData,devData,data = clf.get_data(data)
    #增强数据
    x_train,y_train,x_dev,y_dev,trainData,devData,data = clf.heighten_data(data)
    #训练
    clf.train(x_train,y_train)
    #评估
    score = clf.dev(x_dev,y_dev)
    print(score)
    y_pred = clf.predict(x_dev)
    for i in range(3):
        clf.plot_fit_curve(x_dev,y_dev,y_pred,i=i)
    # parm = {
    #     "n_estimators":[490,500,520],
    #     "max_depth":[28,29,30,31,32],
    #     "min_samples_leaf":[1,2,3]
    # }
    # clf.Cross_Validation(data,parm)
    plt.figure(figsize=(16, 10), dpi = 144)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.scatter(y_pred, y_dev, c='lightskyblue', label='data',alpha=0.8) #训练样本
    # plt.scatter(x_dev.iloc[:, 0], d.predict(x_dev), c='darkorange', label='prediction', lw=2,alpha=0.8) #拟合曲线
    plt.axis('tight')
    plt.title('RandomForestRegressor (k =%f)' % score)
    plt.xlabel("Prices")
    plt.ylabel("Prices")
    plt.show()