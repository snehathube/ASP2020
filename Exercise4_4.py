""" Speeches III """
import pickle
import glob
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

n_features = 1000
n_components = 15
n_top_words = 10
n_topics = 2
count_m = pickle.load(open('./output/speech_matrix.pk', 'rb'))
lda_m = lda(n_components=n_topics, random_state=0)
topics = lda_m.fit_transform(count_m)
files = glob.glob('.\data\speeches\R0*')
corpus = []
text = open('.\data\speeches\R021028A', encoding='utf-8')

for name in files:
       try:
         f = open(name, encoding='utf-8')
         text = f.read()
         corpus.append(text)

       except UnicodeDecodeError:
           print(name)
text = "".join(corpus)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = 'Topic #%d: ' % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print('\n Topics in LDA model:')
#tf_feature_names = count_m.get_feature_names()
#print_top_words(lda_m, tf_feature_names, n_top_words)
topics = pd.DataFrame(topics, columns=["Topic"+str(i+1) for i in range(n_topics)])
print('Created a (%dx%d) document-topic matrix.' % (topics.shape[0], topics.shape[1]))
topics.head()
most_likely_topics = topics.idxmax(axis=1)
most_likely_topics.groupby(most_likely_topics).count()
