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
df_cluster_clean = df_cluster[df_cluster.dbscan>=0]

from sklearn.metrics import silhouette_score
print(silhouette_score(x_scaled, kmeans.labels_))
print(silhouette_score(x_scaled, aggmer.labels_))
print(silhouette_score(x_scaled, dbscan.labels_))
#dbscan has highest silhouette score

iris_df = pd.DataFrame(x, columns=iris['feature_names'])
df_cluster['sepal width'] = iris_df['sepal width (cm)']
df_cluster['petal length'] = iris_df['petal length (cm)']
df_cluster = df_cluster[df_cluster.dbscan>=0]

from matplotlib import pyplot as plt
fig, axes = plt.subplots(1,3, sharey=True, sharex=True)
df_cluster['kmeans'].plot.scatter(df_cluster['sepal width'],df_cluster['petal length'],color='red', ax=axes[0])
df_cluster['aggmer'].plot.scatter(df_cluster['sepal width'],df_cluster['petal length'],color='green', ax=axes[1] )
df_cluster['dbscan'].plot.scatter(df_cluster['sepal width'],df_cluster['petal length'],color='blue', ax=axes[2] )
