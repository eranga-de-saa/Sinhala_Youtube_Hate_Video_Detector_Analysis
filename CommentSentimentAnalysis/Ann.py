import nltk
import random
import json
from pandas.io.json import json_normalize
from pprint import pprint
from collections import defaultdict
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import  MultinomialNB
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import string
import random
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/FYP Comments.xlsx')


df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)

def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train, X_test)


tfidf = TfidfVectorizer()
df['data'] = df['Comment']
print(df.head())
data = tfidf.fit_transform(df['data'])

smote = SMOTE('minority')
X_sm, Y_sm = smote.fit_sample(data, df['sentiment_one_hot'])


X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.2)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train_tfidf.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train_tfidf, y_train,
                              validation_data=(X_test_tfidf, y_test),
                              epochs=20,
                              batch_size=64,
                              verbose=1)

predicted = model.predict(X_test_tfidf)
print(predicted)
