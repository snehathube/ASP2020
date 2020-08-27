""" Unsupervised learning """
# Principal Component Analysis
from sklearn.datasets import load_boston
boston = load_boston()
x=boston['data']

from sklearn.decomposition import PCA
pca=PCA(random_state=0)
pca.fit(x)
print(pca.n_components_)  #underscore in the end means it is something that behaves in a special way. Special here means that this attribtte cannot be accessed before the fitting.
print(pca.explained_variance_)
print(pca.components_) # loadings are seen here

import pandas as pd
df= pd.DataFrame(pca.components_)
df.index += 1  # shorthand for: df.index = df.index + 1; can also do this for -1

import seaborn as sns
sns.heatmap(df, xticklabels=boston['feature_names'])
import matplotlib.pyplot as plt
plt.show()
# careful, there is variance domination. ZN, INDUS, CRIm have high variance and therefore they seem to be more important.
df = pd.DataFrame(x, columns=boston['feature_names'])
df.describe()
# so to get rid of var domination e need to scale the data
#scaling
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x)
print(scaler.data_max_)
print(scaler.data_min_)
x_scaled = scaler.transform(x)
unscaled = pd.DataFrame(x).T  # t frfom transpose
scaled = pd.DataFrame(x_scaled).T  #put scaled and unscaled data in dataframes for comparison
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
sns.scatterplot(x=1, y=3, data=unscaled, ax=ax[0]) #we arae comparing the second and the fourth observation
sns.scatterplot(x=1, y=3, data=scaled, ax=ax[1])
plt.show()

pca = PCA(random_state=0).fit(x_scaled)
sns.heatmap(pca.components_,  xticklabels=boston['feature_names'])
plt.show()
# Now we see much more differneces in colours and loadings
print(boston['DESCR'])

var = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance'])
var.index.name = 'Principal Component'
var['Cum.Explained Variance'] = var['Explained Variance'].cumsum()
var.plot(kind='bar')
plt.show()

x_trans = pca.transform(x)
print(x_trans)
df=pd.DataFrame(x_trans)
df[range(7)].describe() # Cchose first 8 PC's because they explain almost 95% of the data.

# Clustering
#k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x_scaled)
kmeans.labels_
#what you see are clusters

import pandas as pd
df=pd.DataFrame(x,columns=boston['feature_names'])
df['cluster_kmeans'] = kmeans.labels_
df.groupby('cluster_kmeans').mean()
cols = ['INDUS', 'AGE', 'DIS', 'RAD', 'TAX', 'LSTAT']
sns.pairplot(df,vars=cols, hue='cluster_kmeans')
plt.show()

#diagonals give us the kernel density plots
# how do we know we have a good cluster?
# for kmeans a way of evaluation is elbow plot

#Agglomerative clustering is a form of hierarchical clustering
# when you work with text i.e. sparse data use cosine as measure of distance

from sklearn.cluster import AgglomerativeClustering
agg= AgglomerativeClustering(n_clusters=3, affinity='euclid', linkage='complete')
agg.fit(x_scaled)
df['cluster_agg'] = agg.labels_
sns.pairplot(df, vars=cols, hue='cluster_agg', diag_kind='hist')
plt.show()
# Dendogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
linkage_matrix = linkage(x_scaled, metric='euclidean', method= 'complete')
plt.figure(figsize=(10,5))
dendrogram(linkage_matrix)
plt.show()
# On the y axis you have the distance between the points for merging. x axxis has all the data points
df.groupby('cluster_agg').mean()
pd.crosstab(df['cluster_kmeans'], df['cluster_agg'], margins=True)

# Density based spatial clustering of Applications with Noise (DBSCAN)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(min_samples=6, eps=0.65) # you want atleast 3 observatoins around an observation to make this a cluster cohort or the minimize size of a cluster
dbscan.fit(x_scaled)
df['cluster_dbscan'] = dbscan.labels_
df['cluster_dbscan'].value_counts()
# first column shows cluster number while the second shows the size of the cluster.
# -1 shows the number of noise elements i.e. they are not part of any cluster
sns.pairplot(df, vars=cols, hue='cluster_dbscan', diag_kind='hist')
plt.show()
# don't use dbscan for text

#Silhouette score --> for evaluation of how good our clusters from the different methods
# intuition you want high variance between clusters but low within.
# -1 bad and 1 good. take one with the highest SS.
from sklearn.metrics import silhouette_score
print(silhouette_score(x, dbscan.labels_))
print(silhouette_score(x, kmeans.labels_))
print(silhouette_score(x, agg.labels_))

# you would do a grid search with multiple parameters, with k=1,2,3... then put it in a pipeline and then keep the combination that gives the highest SS
for k in range(2,500):
    kmeans= KMeans(n_clusters=k).fit(x_scaled)
    score = silhouette_score(x,kmeans.labels_)
    print(f'k={k}:{score}')
    #for rounding off decimals
    print(f'k={k}:{score:.2}')
# keep k=2
