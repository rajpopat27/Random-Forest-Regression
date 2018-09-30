# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:41:58 2018

@author: RAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)
y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X),max(X),0.01).reshape(-1,1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.show()