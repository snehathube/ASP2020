"""code from day2 of ASP 2020 course"""
# Imports
import pandas as pd
# Main code  create a toy dataframe
d = {'asp':[2020, 2021,2022], 'ifw': [1990, 'x', 2010]}

df = pd.DataFrame(d)
# right click on df --> 'View as dataframe'
# Stop console should be followed by Rerun (which really means restart the kernel)

# overleaf also uses GIT. ny text formats are asy for GIT to check files and compare from.

# Create github account
# add a gitignore to ignore a typical set of file extensions that you would like to ignore. All temp, auxilliary files.It is good for cleanup.

# Machine learning
# ML is more used for prediction rather than causal inference.
# ML ers interested in prediction but economists are more interested in causality/ coefficients.
# pip install sklearn  in the terminal
# k neighbour problem
# distance measure -- Euclidean, Manhattan, cosine
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html?highlight=distance#sklearn.neighbors.DistanceMetric

# Supervised machine learning

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])
x = cancer['data']
y = cancer['target']
y
x
import pandas as pd
df = pd.DataFrame(x, columns=cancer['feature_names'])
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
# by default the train ize is 0.25 of the total sample
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)  # instantiate the model
clf.fit(x_train,y_train) # This is where the magic happens
preds = clf.predict(x_test)  # this is where the magic is used
print(preds[:10])
print(y_train[:10])
clf.score(x_test,y_test)  #accuracy score
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,preds)  #evalutaion matrix
print(confusion_m) #5 are false positives and 7 are false negative

import seaborn as sns
sns.heatmap(confusion_m,annot=True)
import matplotlib.pyplot as plt
plt.show()
#TN(ture negative)   FP
#FN(False negative)  TP
# when you predict all observations to be positive?
# precision becomes one and recall becomes zero.

from sklearn.metrics import classification_report
report = classification_report(y_test,preds,target_names=cancer['target_names'])
print(report)

# Linear models: OLS and Logit
# https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
#  regularization is like sensitivity tests and are used to make sure the model is not over-fitted.

#regression you want to predict prices of houses based on some characteristics
from sklearn.datasets import load_boston
boston =load_boston()
x= boston['data']
y= boston['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
print(boston['DESCR'])
from sklearn.linear_model import Ridge
ridge= Ridge(alpha=0.1).fit(x_train, y_train)
print(ridge.coef_)
# we see the biased coefficients coming from the ridge OLS printed.
#to ease comparison with a lasso we put these coefficients in a dataframe

import pandas as pd
df= pd.DataFrame(ridge.coef_, index=boston['feature_names'], columns=['ridge'])
df

df.plot.bar()
import matplotlib.pyplot as plt
plt.show()

from sklearn.linear_model import Lasso
lasso= Lasso(alpha=0.1).fit(x_train, y_train)
print(lasso.coef_)
df['lasso'] = lasso.coef_
df

df.plot.bar()
import matplotlib.pyplot as plt
plt.show()

#Neural networks
#https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
#Data scaling or normalization is a process of making model data in a standard format so that the training is improved, accurate, and faster. The method of scaling data in neural networks is similar to data normalization in any machine learning problem.
#https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
#For binary classification use sigmoid activation

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])
x = cancer['data']
y = cancer['target']
y
x
import pandas as pd
df = pd.DataFrame(x, columns=cancer['feature_names'])
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(100,100))
mlp.fit(x_train, y_train)
print(mlp.score(x_test,y_test))  #accuracy score

from sklearn.metrics import confusion_matrix
preds = mlp.predict(x_test)
confusion_m = confusion_matrix(y_test,preds)  #evalutaion matrix
print(confusion_m) #5 are false positives and 7 are false negative
import seaborn as sns
sns.heatmap(confusion_m,annot=True)
import matplotlib.pyplot as plt
plt.show()

# try playing with scaling
# Cross-validation  -- test results with different test and training sets of data
# grid search CV
# pipeline

#Grid search with cross validation
from sklearn.datasets import load_boston
boston =load_boston()
x= boston['data']
y= boston['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
pipe = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(max_iter=1000,random_state=0))])
param_grid = {'nn__hidden_layer_sizes': [(10,10), (15,), (15,15)], 'nn__alpha':[0.0001, 0.001, 0.01]}  # two layers with 10nodes each

grid = GridSearchCV(pipe, param_grid, return_train_score=True)
grid.fit(x_train, y_train)
import seaborn as sns
scores = results


