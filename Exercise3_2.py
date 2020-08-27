import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = iris['data']
x_scaled= scaler.fit_transform(x)
#k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x_scaled)

from sklearn.cluster import AgglomerativeClustering
aggmer = AgglomerativeClustering(n_clusters=3)
aggmer.fit(x_scaled)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(min_samples=2, eps=1,metric='euclidean')
dbscan.fit(x_scaled)

df_cluster = pd.DataFrame()
df_cluster['kmeans'] = kmeans.labels_
df_cluster['aggmer'] = aggmer.labels_
df_cluster['dbscan'] = dbscan.labels_

