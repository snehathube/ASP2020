""" Speeches I"""

#a
import glob
from string import digits, punctuation

files = glob.glob('.\data\speeches\R0*')
corpus = []
text_m = open('.\data\speeches\R021028A', encoding='utf-8')
for name in files:
       try:
         f = open(name, encoding="utf-8")
         text_m = f.read()
         corpus.append(text_m)

       except UnicodeDecodeError:
           print(name)

text_m = "".join(corpus)

#b
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

_stemmer = nltk.snowball.SnowballStemmer('english')
def tokenize_and_stem(text_m):
    remove = {p: "" for p in digits + punctuation}
    cleaned = text_m.translate(str.maketrans(remove))
    tokens = nltk.word_tokenize(cleaned.lower())
    text_new = [_stemmer.stem(t) for t in tokens]
    return text_new

_stopwords = nltk.corpus.stopwords.words('english')
_stopwords = tokenize_and_stem(' '.join(_stopwords))

vec = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
mat = vec.fit_transform(corpus)

#c
import pickle
pickle.dump(mat, open('./output/speech_matrix.pk', 'wb'))
