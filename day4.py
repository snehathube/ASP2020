"""Natural Language Processing"""

# weighting algo is tfidf. a word that is used by many documents os something we should put less weight on.
# term frequency, inverse document frequency.
# calculating the linguistic distance between tow documents, so once you have the word vector just take the cosine difference.
# since it is a n*k matrix you can already apply clustering to it.
# pip install nltk
import nltk
nltk.download()

#Vectorization with one text
text='I love you. We love you. Python loves us. Git also loves you.' \
    'This is a noun phrase. 123! Today is a nice day. Best language ever.'
from string import digits, punctuation
punctuation
digits
import nltk
# now we write our first function
# dictionary comprehension is creating a dictionar in one line.

_stemmer = nltk.snowball.SnowballStemmer('english') # like remove last -s or -ing
_stemmer.stem('house')
_stemmer.stem('eating')
_stemmer.stem('household')
#_stemmer = nltk.snowball.SnowballStemmer('german') # like remove last -s or -ing
#_stemmer.stem('Bratwurste')
def tokenize_and_stem(text):
    remove = {p: '' for p in digits + punctuation}
#    remove
    cleaned = text.translate(str.maketrans(remove))
    tokens = nltk.word_tokenize(cleaned.lower())
    new = [_stemmer.stem(t) for t in tokens]
    return new
#equivalen tof the preious line
#new =[]
#for t in tokens:
#    _stemmer.stem()
tokenize_and_stem(text)
_stopwords = nltk.corpus.stopwords.words('english')
_stopwords

from sklearn.feature_extraction.text import CountVectorizer
#max_fd and mimn_df float is share of words in one document but integer is total number of documents.
vec = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
mat = vec.fit_transform([text])
mat.todense().T # saves the coordinates and values of non-stop words

##### Vectorization with corpus #####
#####################################
doc1='I love you. We love you. Python loves us. Git also loves you.' \
    'This is a noun phrase. 123! Today is a nice day. Best language ever.'
doc2=' State is expensive and old. R is strange. Matlab is super expensive.'\
    ' SPSS: WORST language ever. ani.'
corpus=(doc1, doc2)
from string import digits, punctuation
import nltk
_stemmer = nltk.snowball.SnowballStemmer('english') # like remove last -s or -ing
def tokenize_and_stem(text):
    remove = {p: '' for p in digits + punctuation}
    cleaned = text.translate(str.maketrans(remove))
    tokens = nltk.word_tokenize(cleaned.lower())
    new = [_stemmer.stem(t) for t in tokens]
    return new
tokenize_and_stem(text)

_stopwords = nltk.corpus.stopwords.words('english')
_stopwords
#_stopwords.remove('shan') #remove words
#_stopwords.extend(['analyz', 'also']) #add words
_stopwords = tokenize_and_stem(' '.join(_stopwords)) # we tokenize and stem the stop words because some words in the stop words look different before and after eg, any becocmes ani. So with this step we make them exactly how the stopwords would look like in our t and s text
_stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  #max_fd and mimn_df float is share of words in one document but integer is total number of documents.
vec = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
vec = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
mat= vec.fit_transform(corpus)
terms = vec.get_feature_names()

import pandas as pd
cm = pd.DataFrame(mat.todense().T, index=terms)
cm # word ever is used in both documents once...


### Textblob ###
from textblob import TextBlob
blob = TextBlob(corpus[0])
blob.noun_phrases
blob.sentiment #polarity -1 very negative 1 positve, subjectivity:
blob.translate('de') #not  good translator

#Readbility hemingway app

# wordclouds
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc = WordCloud(relative_scaling=1.0, background_color='white').generate(corpus[0])
plt.imshow(wc) # make plot active
plt.show()

#dedupe de-duplicate and find matches in your excel sheet dedupe.io

