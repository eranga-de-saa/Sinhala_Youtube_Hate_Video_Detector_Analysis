from sklearn.externals import joblib
import nltk
import random
import json
from pandas.io.json import json_normalize
from pprint import pprint
from collections import defaultdict
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import sent_tokenize, word_tokenize
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
from sklearn.ensemble import VotingClassifier
import pickle
import os
from random import seed
from random import randint
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import sys
sys.path.insert(0, '../')
from preprocess import preprocessor as pr

comment_sentiModel = joblib.load("../CommentSenti/comment_Lr.pkl")
comment_feature_mapper = joblib.load("../CommentSenti/comment_feature_mapper.pkl")


def main():
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())  # for parallel processing
    df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'posToNegCommentRatio', 'sentiment'])

    i =0
    for x in range(751):
        with open("../../output/"+str(x)+".json", encoding='utf-8') as f:
            data = json.load(f)
            # global commentsText
            commentsText = ""
            # tags = data[0]["tags"]
            sentiment = data[0]['sentiment']
            if 'tags' in data[0].keys():
                tags = str(' '.join(data[0]["tags"]))
                # print("tags" + tags)
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + " " + tags)
            else :
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"])

            likes = int(data[0]["likeCount"])
            dislikes = int(data[0]["dislikeCount"])
            likeDislikeRatio = str(float(likes / dislikes))
            results = pool.map(preprocess, [item for item in data[0]["comments"]])
            postiveCount = 1
            negativeCount = 1
            for result in results:
                if result[0] is not None:
                    commentsText += result[0]
                    if result[1] == 1:
                        postiveCount = postiveCount + 1
                    else:
                        negativeCount = negativeCount + 1
            posToNegCommentRatio = str(float(postiveCount / negativeCount))
            df.loc[i] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [posToNegCommentRatio] + [sentiment]
            print(str(i)+" : "+df['posToNegCommentRatio'].loc[i])

        # print(df['otherMetadata'].iloc[0])

        i += 1

        # Classify
        # Classification_Model = joblib.load("classification_Lr.pkl")
        # Classification_vectorizer  = joblib.load("vectorizer_mapper.pkl")
        #
        # tranformed = Classification_vectorizer.transform(df)
        # print(tranformed[0][2])
        # prediction = Classification_Model.predict(tranformed)
        # print(prediction)
    # print(df)

    df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)

    mapper = DataFrameMapper([
        (['posToNegCommentRatio'], StandardScaler()),
        ('otherMetadata', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        (['likeDislikeRatio'], StandardScaler()),
    ])

    mapper.fit(df)

    features = mapper.transform(df)

    label = df['sentiment_one_hot']

    x, x_test, y, y_test = train_test_split(features, label, test_size=0.2, train_size=0.8, random_state=0)

    clf = LogisticRegression()
    clf.fit(x, y)

    predicted = clf.predict(x_test)

    print(classification_report(y_test, predicted))

    pool.close()


def preprocess(item):
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None):
        transformed = comment_feature_mapper.transform([processed_comment])
        sentiment = int(comment_sentiModel.predict(transformed))
        return processed_comment, sentiment
    else:
        return None, None


if __name__ == '__main__':
    main()
