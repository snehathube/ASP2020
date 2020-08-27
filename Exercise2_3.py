"""Neural Network Regression"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.keys())
x = diabetes['data']
y = diabetes['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
#after Standard-Scaling with in total eight parameter combinations of your choice using 4-fold Cross-Validation.

pipe = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(random_state=0, max_iter=1000,solver='lbfgs', activation='tanh'))])
param_grid = {'nn__hidden_layer_sizes': [(10, 10), (15,15)],
              'nn__alpha': [0.001, 0.00001, 0.001, 0.0002]}
#param_grid = {'nn__hidden_layer_sizes': [(10, 10), (15,15), (20,20), (15,)],
#              'nn__alpha': [0.001, 0.00001]}
#param_grid = {'nn__hidden_layer_sizes': [(5, 5), (10,10), (15,), (15,15)],
#              'nn__alpha': [0.001, 0.0001]}
grid = GridSearchCV(pipe, param_grid, return_train_score=True, cv=4)
grid.fit(x_train, y_train)
print(grid.best_params_)
score = grid.score(x_test, y_test)
print(score)
