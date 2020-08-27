"""Regularization"""
import pandas as pd
from sklearn.model_selection import train_test_split

#a
df_boston_poly = pd.read_csv('./output/polynomials.csv')
#b
y= df_boston_poly['y']
x = df_boston_poly.drop(columns='y')

#df_boston = pd.DataFrame(x, columns=boston['feature_names'])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
#c
from sklearn.linear_model import Ridge
ridge = Ridge().fit(x_train, y_train)
print(ridge.score(x_train,y_train))
print(ridge.coef_)

from sklearn.linear_model import Lasso
lasso = Lasso().fit(x_train, y_train)
print(lasso.score(x_train,y_train))
print(lasso.coef_)
# Error because the number of iterations in default do not allow sufficient convergence
# Accuracy score Ridge = 0.9487427935559346
#                Lasso = 0.9071703674803155

df_coeff = pd.DataFrame(ridge.coef_, index=x.columns,columns=['ridge'])
df_coeff['lasso'] = lasso.coef_

count = df_coeff[(df_coeff['lasso'] == 0) & (df_coeff['ridge'] != 0)].shape
#48

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,30)
df_coeff.plot.barh()
plt.savefig('./output/polynomials.pdf')
