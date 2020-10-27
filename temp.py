# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Veri yükleme

veriler = pd.read_csv("veriler.csv")
# print(veriler)


##Eksik veri işleme

eksikVeriler = pd.read_csv("eksikveriler.csv")

#

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")   ## Ortalamayı yazıcağımızdan dolayı ortalama alma stratejisini alıyoruz

yas = eksikVeriler.iloc[:,1:4].values   ## 1 den 4 e kadar olan satırları alıyoruz

#print(yas)

imputer = imputer.fit(yas[:,1:4])   ## Fit ile öğretiyoruz

yas[:,1:4] = imputer.transform(yas[:,1:4])   ## transfor ile çğrendiğini uygulama

#print(yas)


## ENCODE ETME. KATEGORİGe VERİLERİ NUMERİC YAPMA

ulke = veriler.iloc[:,0:1].values   ## Sadece en baştaki ülkelerini verimizden çekiyoruz

#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()     ## Burda direk tr ye 0 us ye 1 fr ye 2 değerini atıyor numeric yapmak için

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

#print(ulke)

ohe = preprocessing.OneHotEncoder()   ## burda 3 parçaya ayırıyoruz eğer tr ise tr 1 geri kalan 0

ulke = ohe.fit_transform(ulke).toarray()

#print(ulke)


## Şimdi değ,işiklik yaptığımız verileri toplayıp ortak bi veri kümesi elde deicez


sonuc = pd.DataFrame(data=ulke, index= range(22), columns=["fr","tr","us"])

#print(sonuc)


sonuc2 = pd.DataFrame(data = yas, index=range(22),columns = ["boy","kilo","yas"])

#print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values

#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index = range(22),columns = ["cinsiyet"])  

#print(sonuc3)

sDeneme = pd.concat([sonuc,sonuc2])   ## dataları alt alta ekliyor bu şekilde

#print (sDeneme)


s = pd.concat([sonuc,sonuc2],axis=1)  ## Yan yana dataları birleştiriyor

#print(s)

s2 = pd.concat([s,sonuc3],axis=1)  ## En son tüm düzeltilmiş datalarımızı aldık ve birleştirdik.

print(s2)

# GELEN VERİYİ BÖLEREK AYRI AYRI ÇALIŞMA



from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.50,random_state=0) ## test ve train data seti oluşturutoruz. 4 farklı parçaya bölüyoruz. yarısı random bi şekilde teste yarısı dataya



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()    ## Öznitelik ölçekleme, farklı dünyadaki veirlerin aynı dünyaya çekişimizi gösterdik.

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)




















