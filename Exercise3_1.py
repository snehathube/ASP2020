"""Principal Component Analysis"""
import pandas as pd
df_oly = pd.read_csv('./data/olympics.csv' , index_col=0)
df_oly.describe()
# Score has a higher variance than all the other features therefore it might misleadingly come out as the prominent feature in PCA because of variance domination.
# Probably scaling the data is necessary before we proceed.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = df_oly.values
x_scaled = scaler.fit_transform(x)
df_oly_scaled = pd.DataFrame(x_scaled)
df_oly_scaled.describe()
from sklearn.decomposition import PCA
pca=PCA()
pca = PCA(random_state=0).fit(x_scaled)
df_pca = pd.DataFrame(pca.components_,columns=df_oly.columns)
# 110 is more prominent is first components
# disq in the second
# haut in the third

import matplotlib.pyplot as plt
var = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance'])
var.index.name = 'Principal Component'
import numpy as np
var['Cum.Explained Variance'] = np.cumsum(var)
var.plot(kind='bar')
plt.show()

# For atleast 90% vaiance (without rounding off) we need 6 components.