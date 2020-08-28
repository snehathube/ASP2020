""" Neural Networks Classifier"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#a
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

import pandas as pd
df = pd.DataFrame(x, columns=cancer['feature_names'])
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)

#b
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
scaler =MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf= MLPClassifier(random_state=0,max_iter=1000)
