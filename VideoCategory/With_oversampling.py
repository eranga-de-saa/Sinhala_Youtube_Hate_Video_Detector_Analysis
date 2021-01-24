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

dataset = defaultdict(list)

# create a dataframe to store dataset
df = pd.DataFrame(columns=['comment', 'replies', 'otherMetadata', 'likeDislikeRatio', 'category'])


i =0
for x in range(603):
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
# class variable binerization

# plt.figure(figsize=(10, 4))
# df['category'].value_counts().plot(kind='bar')
# plt.show()


cvec = CountVectorizer()
df['data'] = df['comment'] + df['otherMetadata']
print(df.head())
data = cvec.fit_transform(df['data'])

# Synthetic Minority Over-sampling Technique increased accuracy by 3%
smote = SMOTE('minority')
X_sm,Y_sm = smote.fit_sample(data, df['category'])

X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.2)

nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=df['category'].unique()))


# # unbalanced data set


# df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)



