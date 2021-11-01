# _*_ coding utf-8 _*_
# @Time     : 2021/6/17 17:05
# @Author   : yaocctao
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierByUDF():
    def __init__(self):
        self.rfc = RandomForestClassifier(n_estimators=150,criterion="entropy",max_depth=20)

    def get_data(self,data,frac=0.8):
        data = data[['Direction', 'District', 'Garden', 'Region', 'Floor',
                     'Renovation', 'Size', 'Year', 'Room', 'Elevator', 'Price','Hall']]
        trainData = data.sample(frac=frac, random_state=0, axis=0)
        x_train = trainData.iloc[:, :-1]
        y_train = trainData["Hall"]
        devData = data[~data.index.isin(trainData.index)]
        x_dev = devData.iloc[:, :-1]
        y_dev = devData["Hall"]

        return x_train,y_train,x_dev,y_dev,trainData,devData,data

    def fit(self,x_train,y_train,x_dev,y_dev):
        self.rfc.fit(x_train,y_train)
        print("score:",self.rfc.score(x_dev,y_dev))

    def add_missing_hall(self):
        datanull = pd.read_csv("./data/data_clean.csv")
        encodedata = pd.read_csv("./data/encode_data.csv")
        Traindata = encodedata[~encodedata.index.isin(datanull[datanull["Hall"].isnull()].index)]
        Traindata = Traindata[['Direction', 'District', 'Garden', 'Region', 'Floor',
                               'Renovation', 'Size', 'Year', 'Room', 'Elevator','Price', 'Hall']]
        testdata = encodedata[encodedata.index.isin(datanull[datanull["Hall"].isnull()].index)]
        self.rfc.fit(Traindata.iloc[:,:-1],Traindata["Hall"])
        encodedata.loc[encodedata.index.isin(datanull[datanull["Hall"].isnull()].index),'Hall'] = self.rfc.predict(testdata.iloc[:,:-1])
        encodedata.to_csv("./data/encode_data_addMissHall.csv",index=False)

if __name__ == '__main__':
    #实例化模型
    model = RandomForestClassifierByUDF()
    #读取数据
    data = pd.read_csv("./data/encode_data.csv")
    x_train,y_train,x_dev,y_dev,trainData,devData,data = model.get_data(data)
    #训练模型
    model.fit(x_train,y_train,x_dev,y_dev)
    #填写缺失值
    model.add_missing_hall()

