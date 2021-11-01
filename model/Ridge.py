# _*_ coding utf-8 _*_
# @Time     : 2021/6/28 15:38
# @Author   : yaocctao
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

#获取数据函数分成训练验证集
class RidgeMoedl():
    def __init__(self):
        self.Ridge =Ridge()

    def get_data(self,data,frac=0.8):
        trainData = data.sample(frac=frac, random_state=0, axis=0)
        x_train = trainData.iloc[:, :-1]
        y_train = trainData["Price"]
        devData = data[~data.index.isin(trainData.index)]
        x_dev = devData.iloc[:, :-1]
        y_dev = devData["Price"]

        return x_train,y_train,x_dev,y_dev,trainData,devData,data

    def pipeline(self,degree=3):
        self.r = make_pipeline(PolynomialFeatures(degree),self.Ridge)
        #训练
        start = time.time()
        self.r.fit(x_train,y_train)
        print(self.r.score(x_dev,y_dev))
        end = time.time()
        print("time:"+str(end-start))

    def plot(self):
        plt.figure(figsize=(16, 10), dpi = 144)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        plt.scatter(x_dev.iloc[:, 0], y_dev, c='b', label='data', s=100) #训练样本
        plt.scatter(x_dev.iloc[:, 0], self.r.predict(x_dev), c='darkorange', label='prediction', lw=2) #拟合曲线
        plt.axis('tight')
        plt.title('Ridge regression（Polynomial） (k =%f)' % self.r.score(x_dev,y_dev))
        plt.ylabel("Prices")
        plt.show()


if __name__ == '__main__':
    #读取数据
    data = pd.read_csv("./data/data_final.csv")
    #实例话模型
    data = pd.concat([data[["Size","Year","Floor","Renovation"]],data.iloc[:, 70:93],data["Price"]],axis=1)
    model = RidgeMoedl()
    #生成训练集和验证集
    x_train,y_train,x_dev,y_dev,trainData,devData,data = model.get_data(data)
    #进行3此多项式插值
    model.pipeline(degree=3)
    #画图
    model.plot()