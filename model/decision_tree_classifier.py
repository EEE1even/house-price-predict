# _*_ coding utf-8 _*_
# @Time     : 2021/6/28 16:20
# @Author   : yaocctao
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DecisionTreeClassifierByUDF():
    def __init__(self):
        self.dtc = DecisionTreeClassifier(criterion="entropy")

    def get_data(self,data,frac=0.8):
        data = data[['Direction', 'District', 'Garden', 'Region', 'Floor',
                                'Renovation', 'Size', 'Year', 'Room', 'Elevator','Price', 'Hall']]
        trainData = data.sample(frac=frac, random_state=0, axis=0)
        x_train = trainData.iloc[:, :-1]
        y_train = trainData["Hall"]
        devData = data[~data.index.isin(trainData.index)]
        x_dev = devData.iloc[:, :-1]
        y_dev = devData["Hall"]

        return x_train,y_train,x_dev,y_dev,trainData,devData,data

    def fit(self,x_train,y_train,x_dev,y_dev):
        self.dtc.fit(x_train,y_train)
        print(self.dtc.score(x_dev,y_dev))


if __name__ == '__main__':
    #实例化模型
    model = DecisionTreeClassifierByUDF()
    #读取数据
    data = pd.read_csv("./data/encode_data.csv")
    x_train,y_train,x_dev,y_dev,trainData,devData,data = model.get_data(data,frac=0.8)
    #训练
    model.fit(x_train,y_train,x_dev,y_dev)