# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:21:08 2021

@author: Erik Trincado
"""
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Lectura de base de datos
df = pd.read_csv("train.csv")

#Convertir variables categoricas a numericas y crear nuevas filas
df["Sex_cleaned"] = np.where(df["Sex"]=="male",0,1)
df["Embarked_cleaned"] = np.where(df["Embarked"]=="S",0,
                                  np.where(df["Embarked"]=="C",1,
                                  np.where(df["Embarked"]=="Q",2,3)))

#Eliminar valores en NaN
df = df[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"]].dropna(axis=0, how='any')

#Definir variables para algoritmo
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"]

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis()
X = df[used_features]
y = df["Survived"]
x_lda = lda_model.fit_transform(X,y)
x_train, x_test, y_train, y_test= train_test_split(x_lda,y,random_state=1)
data = DecisionTreeClassifier()
data.fit(x_train, y_train)
y_pred = data.predict(x_test)
print("MATRIZ LDA")
print(confusion_matrix(y_test, y_pred))
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          y_test.shape[0],
          (y_pred != y_test).sum(),
          100*(1-(y_pred != y_test).sum()/y_test.shape[0])))

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_model = QuadraticDiscriminantAnalysis()
X = df[used_features]
y = df["Survived"]
x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=1)
y_pred = qda_model.fit(x_train,y_train).predict(x_test)
print("MATRIZ QDA")
print(confusion_matrix(y_test, y_pred))
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          y_test.shape[0],
          (y_pred != y_test).sum(),
          100*(1-(y_pred != y_test).sum()/y_test.shape[0])))
