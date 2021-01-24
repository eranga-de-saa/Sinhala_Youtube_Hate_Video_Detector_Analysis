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
# dataset = defaultdict(list)
import sys
sys.path.insert(0, '../')
from preprocess import preprocessor as pr
from sklearn.utils import shuffle


def main():

    df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/FYP Comments.xlsx')
    # print (df)

    # print(df.iloc[298])
    # for index, row in df.iterrows():
    #
    #     df['preprocessed_text'].loc[index] = preprocess(df['Comment'].iloc[index])

    df['preprocessed_text'] = df.apply(lambda x: preprocess(x["Comment"]), axis=1)

    df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)

    countVec_comment = CountVectorizer()


    label = df['sentiment_one_hot']

    # X_train, X_test, y_train, y_test = train_test_split(df['Comment'], label, test_size=0.2, train_size=0.8, random_state=0)

    # df = shuffle(df)
    # traindf=df[:1200]
    # testdf = df[1200:1520]

    traindf, testdf = train_test_split(df, test_size=0.2)

    countVec_comment.fit(traindf['preprocessed_text'])

    positive_score_comment = train(traindf,  countVec_comment)

    prediction = predict(testdf, positive_score_comment)

    print(classification_report(testdf['sentiment_one_hot'], prediction))

    # Comment lexical analysis
    # print(df.head())



def preprocess(item):
    comment = str(item)
    processed_comment = pr.process(comment)
     # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None):
        # commentsText += processed_comment
        return processed_comment
    else:
        return None


def train(traindf, countVec_comment):

    neg_doc_matrix_comment = countVec_comment.transform(traindf[traindf['sentiment_one_hot'] == 0]['preprocessed_text'])
    pos_doc_matrix_comment = countVec_comment.transform(traindf[traindf['sentiment_one_hot'] == 1]['preprocessed_text'])
    neg_tf_comment = np.sum(neg_doc_matrix_comment, axis=0)
    pos_tf_comment = np.sum(pos_doc_matrix_comment, axis=0)
    neg_comment = np.squeeze(np.asarray(neg_tf_comment))
    pos_comment = np.squeeze(np.asarray(pos_tf_comment))
    term_freq_df_comment = pd.DataFrame([neg_comment, pos_comment], columns=countVec_comment.get_feature_names()).transpose()
    term_freq_df_comment.columns = ['negative', 'positive']
    term_freq_df_comment['total'] = term_freq_df_comment['negative'] + term_freq_df_comment['positive']
    term_freq_df_comment['positive_sent_score'] = term_freq_df_comment['positive'] / term_freq_df_comment['total']
    term_freq_df_comment['negative_sent_score'] = term_freq_df_comment['negative'] / term_freq_df_comment['total']
    term_freq_df_comment['polarity_score'] = 2*term_freq_df_comment['positive_sent_score'] - 1
    term_freq_df_comment = term_freq_df_comment.drop(term_freq_df_comment[(term_freq_df_comment['positive_sent_score'] < 0.6) & (term_freq_df_comment['positive_sent_score'] > 0.4)].index)
    # print(term_freq_df[term_freq_df['polarity_score'] < -0.5].sort_values(by='polarity_score', ascending=False).iloc[:15])
    # print(term_freq_df.columns)
    # print(term_freq_df)
    positive_score_comment = term_freq_df_comment.positive_sent_score
    # print(term_freq_df['total']['යන'])
    # print(positive_score_comment)
    # val_probability = []
    # print(positive_score["යන"])

    #     # for w in doc.split():
    #     #     print (w)


    # joblib.dump(positive_score_comment, "positive_score_comment_pickle.pkl")
    return positive_score_comment

def predict(testdf,positive_score_comment):
    testdf = testdf.reset_index()
    print(testdf)

    prediction = []
    for index, row in testdf.iterrows():
        print(str(index))
        # comment automatic annotation
        print(testdf['preprocessed_text'].iloc[index])
        sentiment_score_comment = [positive_score_comment[w] for w in testdf['preprocessed_text'].iloc[index].split() if w in positive_score_comment.index]

        if len(sentiment_score_comment) > 0:
            prob_score_comment = np.mean(sentiment_score_comment)
        else:
            prob_score_comment = np.random.random()

        prediction_comment = 1 if prob_score_comment > 0.5 else 0

        prediction.append(prediction_comment)
    return prediction






if __name__ == '__main__':
    main()