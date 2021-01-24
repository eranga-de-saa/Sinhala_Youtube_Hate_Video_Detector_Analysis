from NaiveBayesClassifier import NaiveBayesClassifier
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
# from sklearn.feature_extraction.text import TfidfVectorizer
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

df_positive_words =  pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\positivewords.xlsx')

df_negative_words=  pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\HateWords.xlsx')
# print (df)

# print(df_negative_words.head())
#
# print(df_positive_words.head())

# print(df_positive_words.keys())

positive_vocabulary = [word.strip() for word in df_positive_words['words']]
negative_vocabulary = [word.strip() for word in df_negative_words['words']]

def main():
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())  # for parallel processing
    df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'sentiment'])

    i =0
    for x in range(751):
        with open("../output/"+str(x)+".json", encoding='utf-8') as f:
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
            for result in results:
                if result is not None:
                    commentsText += result[0]
            df.loc[i] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [sentiment]
            # print(str(i)+" : "+df['posToNegCommentRatio'].loc[i])

        # print(df['otherMetadata'].iloc[0])
        print(i)
        i += 1

    df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)

    df['data'] = df['comment'] + ' ' + df['otherMetadata']

    traindf, testdf = train_test_split(df, test_size=0.2)

    x_train, x_test, y_train, y_test = train_test_split(df['data'], df['sentiment_one_hot'], test_size=0.2)

    NBModel = NaiveBayesClassifier()
    NBModel.train(x_train, y_train, alpha=1)

    print(y_test)
    # hateVideoComments = df.loc[18]['comment']

    # print(hateVideoComments)
    levelOfHate = NBModel.getHateLevel(x_test)
    print(levelOfHate)



def preprocess(item):
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None) and (processed_comment != ""):
        return processed_comment
    else:
        return None




if __name__ == '__main__':
    main()
