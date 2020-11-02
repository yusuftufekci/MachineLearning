#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:43:07 2020

@author: yusuftufekci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:06:37 2020

@author: yusuftufekci
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

veriler =pd.read_excel("iris.xls")

x = veriler.iloc[:,0:4].values ##bağımsız değişkenler
y = veriler.iloc[:,4:].values  ## bağımlı değişkenler

##Eğitim için verilerin bölünmesi

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33, random_state=0) ##test_size train ve test datalarını sayısını belirlemek için kullanılıyor

##Verileri scale ediyoruz
sc = StandardScaler()

X_train = sc.fit_transform(x_train)   ## MODELİ oluşturmak için kullangımız datalar (train) SCALER ŞEKLİNDE
X_test = sc.transform(x_test)


##Logistic regression

from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(random_state=0)

logR.fit(X_train,y_train)  ##Train

y_pred = logR.predict(X_test)   ##Predict


##Confusion Matrix ile başarımıza bakıyoruz
print("logistic regression")
cm = confusion_matrix(y_test,y_pred)

print(cm)


##KNN algoritması

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")

knn.fit(X_train,y_train)

y_pred2=knn.predict(X_test)

print("KNN")
cm = confusion_matrix(y_test,y_pred2)
print(cm)


##SVC algoritması // Support Vector Machine/classifier
from sklearn.svm import SVC

svc = SVC(kernel="linear")
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred3)
print("SVC")
print(cm)



from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
##Bernoulli for naive bayes
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
print("Bernouilli")
y_pred5 = bnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred5)
print(cm)



gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred4 = gnb.predict(X_test)

##Gaussian for naive bayes
print("Gaussian")
cm = confusion_matrix(y_test,y_pred4)
print(cm)

##Decision Tree algoritması

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy") ##Entropy modelini kullandık farklı modellerde var
dtc.fit(X_train,y_train)
y_pred6 = dtc.predict(X_test)

print("Decision tree")

cm = confusion_matrix(y_test,y_pred6)
print(cm)


##Random Forrest Algoritması
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,criterion="entropy") ## parametrelerle oynarak istediğimiz şekle getirebiliriz.
rfc.fit(X_train,y_train)
y_pred7 = rfc.predict(X_test)

print("RFC")
cm = confusion_matrix(y_test,y_pred7)
print(cm)

from sklearn import metrics
y_proba = rfc.predict_proba(X_test) ##Bu bizim olasılıklarımız.
print(y_proba[:,0]) ## ilk baştaki değerin olasılıklarını bastırıyorum
#ROC hesaplaması gibi

fpr , tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label="e")















