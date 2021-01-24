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
# from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import string
import random
from sklearn.ensemble import VotingClassifier
import pickle
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


dataset = defaultdict(list)

# create a dataframe to store dataset
df = pd.DataFrame(columns=['comment', 'replies', 'otherMetadata', 'likeDislikeRatio', 'category'])


i =0
for x in range(751):
    with open("../output/"+str(x)+".json", encoding='utf-8') as f:
        data = json.load(f)
    commentsText = ""
    replyText=""
    replies = defaultdict(list)
# store sentiment of the data
    category = data[0]['category']
    otherMetaData = data[0]["title"] + data[0]["description"]  # + item["tags"]
    likes = int(data[0]["likeCount"])
    dislikes = int(data[0]["dislikeCount"])
    likeDislikeRatio = str(float(likes/dislikes))
    for item in data[0]["comments"]:
        commentsText += item["comment"]
        replyText = ""
        if 'replies' in item.keys():
            for reply in item['replies']:
                replyText += reply['replyComment']
    df.loc[i] = [commentsText] + [replyText] + [otherMetaData] + [likeDislikeRatio] + [category]
    i += 1


def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train, X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    model = Sequential()
    node = 512 # number of nodes
    nLayers = 4 # number of  hidden layer
    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0, nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', # sparse_categorical_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model




tfidf = TfidfVectorizer()
df['data'] = df['comment'] + df['otherMetadata']
print(df.head())
data = tfidf.fit_transform(df['data'])

smote = SMOTE('minority')
X_sm, Y_sm = smote.fit_sample(data, df['category'])


encoder = LabelEncoder()
encoder.fit(Y_sm)
encoded_Y = encoder.transform(Y_sm)
# convert integers to dummy variables
dummy_y = np_utils.to_categorical(encoded_Y)

print(dummy_y)

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_sm, dummy_y, test_size=0.2)

# X_train_tfidf, X_test_tfidf = TFIDF(X_train, X_test)
model_DNN = Build_Model_DNN_Text(X_train_tfidf.shape[1], 4)
model_DNN.fit(X_train_tfidf, y_train,
                              validation_data=(X_test_tfidf, y_test),
                              epochs=20,
                              batch_size=64,
                              verbose=1)
predicted = model_DNN.predict(X_test_tfidf)
y_pred_vector=np.argmax(predicted, axis=1)
y_test_vector=np.argmax(y_test, axis=1)
print(classification_report(y_test_vector, y_pred_vector))


# Synthetic Minority Over-sampling Technique increased accuracy by 3%
# smote = SMOTE('minority')

# max_words = 5000 num_words=max_words

# tokenizer_obj = Tokenizer()
# tokenizer_obj.fit_on_texts(df['data'])
#
# X_train, X_test, y_train, y_test = train_test_split(df['data'], df['category'], test_size=0.2)
#
# X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
# X_test_Tokens = tokenizer_obj.texts_to_sequences(X_test)
#
# # data = tokenizer_obj.texts_to_sequences(df['data'])
# # data = tokenizer_obj.sequences_to_matrix(df['data'], mode='binary')
# # print(data)
#
# max_length = max([len(s.split())] for s in df['data'])
# print(max_length)
# vocab_size = len(tokenizer_obj.word_index) + 1
#
# # print(data.shape())
#
# # X_sm, Y_sm = smote.fit_sample(data, df['category'])
#
# print('working ...')
#
# # X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.2)
#
#
# X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
# X_test_pad = pad_sequences(X_test_Tokens, maxlen=max_length, padding='post')
#
# EMBEDDING_DIM = 100
# print('build model ...')
#
# model = Sequential()
# model.add(Embedding(vocab_size,EMBEDDING_DIM,input_length=max_length))
# model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(4, activation='sigmoid'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# print('train...')
# model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)
#
