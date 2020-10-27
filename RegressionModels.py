#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:29:29 2020

@author: yusuftufekci
"""

##Polinomial regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##from sklearn.cross_validation import train_test_split
##VERİ YÜKLEME
veriler =pd.read_csv("maaslar.csv")


## data fram oluşturma.(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
##Numpy array dönüşümü
X=x.values
Y=y.values

##Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x.values,y.values)



##Polynomial regressiona bakıyoruz burda
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(X)
y_poly = poly_reg.fit_transform(Y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)



##degree arttırıyoruz

poly_reg3 = PolynomialFeatures(degree=4)

x_poly3 = poly_reg3.fit_transform(X)
y_poly = poly_reg3.fit_transform(Y)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)




##görselleştirme
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.show()


plt.scatter(X,Y,color="blue")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="red") ## Normal linear regression yapar gibi yapıyorsun ama önemli nokta şu ki verdiğin değeri polynomial features kullanarak önce bi polynomial hale getirmek gerekiyor
plt.show()

plt.scatter(X,Y,color="blue")
plt.plot(X,lin_reg3.predict(poly_reg.fit_transform(X)),color="red") ## Normal linear regression yapar gibi yapıyorsun ama önemli nokta şu ki verdiğin değeri polynomial features kullanarak önce bi polynomial hale getirmek gerekiyor
plt.show()

print(lin_reg.predict([[11]]))


print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


## SVM SUPPORT VECTOR MACHİNE ALGOR

## scaling muhabbeti önemli. kernel fonksiyonu olarak 4 farklı elimizde fonkyion var onları verilere göre kullanabiliriz.
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()    ## Öznitelik ölçekleme, farklı dünyadaki veirlerin aynı dünyaya çekişimizi gösterdik.

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()

y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel= "rbf")

svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")

plt.show()

print(svr_reg.predict([[11]]))

##Decision tree ile regression yapma

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)


r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")

plt.plot(X,r_dt.predict(X),color="blue")

print(r_dt.predict([[11]]))

plt.show()

##Rassan ağaçlar

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))


plt.scatter(X,Y,color="red")

plt.plot(X,rf_reg.predict(X),color="green")


























 