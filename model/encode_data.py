# _*_ coding utf-8 _*_
# @Time     : 2021/6/28 15:50
# @Author   : yaocctao
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_data(data):
    df = data
    data_numbers = df.select_dtypes(include=['int64','float64'])
    df = df.select_dtypes(include=['object'])
    data = pd.DataFrame()
    for i in range(len(df.columns)):
        l = LabelEncoder()
        l.fit(df.iloc[:,i])
        label = l.transform(df.iloc[:,i])
        label = pd.Series(label,name=df.columns[i])
        data = pd.concat([data,label],axis=1)
    data =pd.concat([data,data_numbers],axis=1)
    return data
if __name__ == '__main__':
    data = pd.read_csv("./data/data_final.csv")
    data = data.drop(data.columns[11:-3].append(pd.Index(["Layout","Id"])),axis=1)
    data = encode_data(data)
    print(data.columns)
    data = data[['Direction', 'District', 'Garden', 'Region', 'Floor',
                 'Renovation', 'Size', 'Year', 'Room', 'Elevator', 'Hall','Price']]
    data.to_csv("./data/encode_data.csv",index=False)
