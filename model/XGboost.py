# _*_ coding utf-8 _*_
# @Time     : 2021/6/30 18:37
# @Author   : yaocctao
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def get_data(data,frac=0.8):
    data = data[['Size', 'Year', 'Room','Direction','District','Garden', 'Region', 'Floor',
                 'Renovation', 'Elevator', 'Hall','Price']]
    trainData = data.sample(frac=frac, random_state=0, axis=0)
    x_train = trainData.iloc[:, :-1]
    y_train = trainData["Price"]
    devData = data[~data.index.isin(trainData.index)]
    x_dev = devData.iloc[:, :-1]
    y_dev = devData["Price"]

    return x_train,y_train,x_dev,y_dev,trainData,devData,data

xgb_n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]

xgb_max_depth = [int(x) for x in np.linspace(2, 20, 10)]

xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]

xgb_tree_method = ['auto', 'exact', 'approx', 'hist', 'gpu_hist']

xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]

xgb_gamma = [int(x) for x in np.linspace(0, 0.5, 6)]

xgb_objective = ['reg:squarederror', 'reg:squaredlogerror']

xgb_grid = {'n_estimators': xgb_n_estimators,
            'max_depth': xgb_max_depth,
            'min_child_weight': xgb_min_child_weight,
            'tree_method': xgb_tree_method,
            'eta': xgb_eta,
            'gamma': xgb_gamma,
            'objective': xgb_objective}

data = pd.read_csv("./data/encode_data_addMissHall.csv")
x_train,y_train,x_dev,y_dev,trainData,devData,data = get_data(data)

xgb_base = XGBRegressor()
xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = xgb_grid,
                                n_iter = 200, cv = 3, verbose = 2,
                                random_state = 420, n_jobs = -1)

xgb_random.fit(x_train,y_train)

print(xgb_random.best_params_)

x = XGBRegressor(tree_method= 'auto', objective= 'reg:squarederror', n_estimators= 1400, min_child_weight= 10, max_depth= 6, gamma= 0, eta= 0.1)
bst = x.fit(x_train,y_train)
print(bst.score(x_dev,y_dev))
y_pred = x.predict(x_dev)
print(x.get_xgb_params())

plt.figure(figsize=(16, 10), dpi = 144)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.scatter(y_pred, y_dev, c='lightskyblue', label='data',alpha=0.8) #训练样本
plt.axis('tight')
plt.title('RandomForestRegressor (k =%f)' % bst.score(x_dev,y_dev))
plt.xlabel("Prices")
plt.ylabel("Prices")
plt.show()