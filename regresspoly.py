# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:44:49 2018

@author: therock
"""

import numpy as np
import pandas as pd


dataset=pd.read_csv('Position_Salaries.csv') 
X=dataset.iloc[:,[1]].values
Y=dataset.iloc[:,[2]].values

#linear regression
from sklearn.linear_model import LinearRegression
lregres=LinearRegression()
lregres.fit(X,Y)

#polynomial regeression
from sklearn.preprocessing import PolynomialFeatures
polyregres=PolynomialFeatures(degree=4)
xpoly=polyregres.fit_transform(X)
polregres=LinearRegression()
polregres.fit(xpoly,Y)

#visual of regression model

import matplotlib.pyplot as mp
mp.scatter(X ,Y , color='red')
mp.plot(X,lregres.predict(X),color='blue')
mp.title('truth or bluff(Linear Regression)')
mp.xlabel('position level')
mp.ylabel('salary')
mp.show()

#visual of polynomial regression model
import matplotlib.pyplot as mp
mp.scatter(X ,Y , color='red')
mp.plot(X,polregres.predict(xpoly),color='blue')
mp.title('truth or bluff(Polynomial Regression)')
mp.xlabel('position level')
mp.ylabel('salary')
mp.show()
