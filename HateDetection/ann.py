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
df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'sentiment'])


i =0
for x in range(751):
    with open("../output/"+str(x)+".json", encoding='utf-8') as f:
        data = json.load(f)
    commentsText = ""
    replyText=""
    replies = defaultdict(list)
# store sentiment of the data
    sentiment = data[0]['sentiment']
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
    df.loc[i] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [sentiment]
    i += 1
# class variable binerization

df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)

mapper = DataFrameMapper([
    ('comment', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
    ('otherMetadata', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
    (['likeDislikeRatio'], StandardScaler()),
])

mapper.fit(df)


features = mapper.transform(df)
print("Shape .......")
print(features.shape)
# joblib.dump(mapper, "vectorizer_mapper.pkl")

label = df['sentiment_one_hot']

x, x_test, y, y_test = train_test_split(features, label, test_size=0.2,train_size=0.8, random_state = 0)


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# print(classification_report(y_test_vector, y_pred_vector))


model.fit(x, y, validation_data=(x_test, y_test),
                              epochs=20,
                              batch_size=64,
                              verbose=1)



# Comment lexical analysis
# print(df.head())

# trans_table = {ord(c): None for c in string.punctuation + string.digits}
#
# def tokenize(text):
#     # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
#     tokens = [word for word in nltk.word_tokenize(text.translate(trans_table))]
#     return tokens
#
#
# tfidf_vectorizer_comments = TfidfVectorizer(tokenizer=tokenize)
# tfidf_vectorizer_comments.fit(df['comment'])
# df['tf_idf_comments'] = tfidf_vectorizer_comments.transform(df['comment']).toarray()
#
# tfidf_vectorizer_replies = TfidfVectorizer(tokenizer=tokenize)
# tfidf_vectorizer_replies.fit(df['replies'])
# df['tf_idf_replies'] = tfidf_vectorizer_replies.transform(df['replies']).toarray()
#
# tfidf_vectorizer_other = TfidfVectorizer(tokenizer=tokenize)
# tfidf_vectorizer_other.fit(df['otherMetadata'])
# df['tf_idf_other'] = tfidf_vectorizer_other.transform(df['otherMetadata']).toarray()






# count_vectorizer_comments = CountVectorizer(tokenizer=tokenize)
# count_vectorizer_comments.fit(df['comment'])
# df['tf_idf_comments'] =count_vectorizer_comments.transform(df['comment'])
#
# count_vectorizer_replies = CountVectorizer(tokenizer=tokenize)
# count_vectorizer_replies.fit(df['replies'])
# df['tf_idf_replies'] =count_vectorizer_replies.transform(df['replies'])
#
# count_vectorizer_other = CountVectorizer(tokenizer=tokenize)
# count_vectorizer_other.fit(df['otherMetadata'])
# df['tf_idf_other'] =count_vectorizer_other.transform(df['otherMetadata'])




# model_data = df[['tf_idf_comments', 'tf_idf_replies', 'tf_idf_other', 'sentiment_one_hot']]
# X = model_data[['tf_idf_comments', 'tf_idf_replies', 'tf_idf_other']]
# Y = model_data['sentiment_one_hot']
# model = LogisticRegression()
# # print(model_data)
# # print(df['tf_idf_other'])
# pd.set_option('display.expand_frame_repr', False)
# print(str(model_data['tf_idf_other'].iloc[1]))
# print(len(X))
# print(len(Y))
# # model.fit(X, Y)


# print (df['tf_idf_comments'].iloc[0])
# print(len(countVec.get_feature_names()))

