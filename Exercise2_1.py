"""Feature Engineering"""
import pandas as pd
#a
from sklearn.datasets import load_boston
boston = load_boston()
x = boston['data']
df_boston = pd.DataFrame(x, columns=boston['feature_names'])

#b
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, include_bias=False)
poly_boston = poly.fit_transform(x)
print(poly_boston.shape)
#104

poly_features= poly.get_feature_names(boston['feature_names'])
df_boston_poly = pd.DataFrame(poly_boston, columns=poly_features)
df_boston_poly['y'] = boston['target']
df_boston_poly.to_csv('./output/polynomials.csv')
