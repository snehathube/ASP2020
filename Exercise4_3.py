"""Speeches II"""
#a
import pickle
cm = pickle.load(open('./output/speech_matrix.pk', 'rb'))

#b
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import csr_matrix

arr_cm = csr_matrix.toarray(cm)
linkage_m = linkage(arr_cm, metric="cosine", method="complete")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off')       # ticks along the bottom edge are off
speeches_dendrogram = dendrogram(linkage_m)
plt.plot()
plt.show()
#d
plt.savefig("./output/speeches_dendrogram.pdf")

