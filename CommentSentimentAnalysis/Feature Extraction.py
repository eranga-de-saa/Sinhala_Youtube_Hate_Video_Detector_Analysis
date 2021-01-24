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
import pickle
from sklearn.externals import joblib


import sys
sys.path.insert(0, '../')
from preprocess import preprocessor as pr


dataset = defaultdict(list)

# create a dataframe to store dataset

def preprocess(item):
    comment = str(item)
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None):
        # commentsText += processed_comment
        return processed_comment
    else:
        return None

df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/FYP Comments.xlsx')

df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)

# for index, row in df.iterrows():
#
#     df['preprocessed_text'].loc[index] = preprocess(df['Comment'].iloc[index])

df['preprocessed_text'] = df.apply(lambda x: preprocess(x["Comment"]), axis=1)


hate_word_list = []
not_hate_word_list = []

# comment_feature_mapper = TfidfVectorizer(ngram_range=(1, 3))
#
# features = comment_feature_mapper.fit_transform(df['preprocessed_text'])

# tfidf_positive_comment = TfidfVectorizer(ngram_range=(1, 3), max_features=100)
# tfidf_negative_comment = TfidfVectorizer(ngram_range=(1, 3), max_features=100)

# neg_word_matrix = tfidf_negative_comment.fit_transform(df[df['sentiment_one_hot'] == 0]['Comment'])
# pos_word_matrix = tfidf_positive_comment.fit_transform(df[df['sentiment_one_hot'] == 1]['Comment'])
#
countVec_comment = CountVectorizer()
# print(df['preprocessed_text'].head())
countVec_comment.fit(df['preprocessed_text'])
# print(countVec_comment.get_feature_names())

# print(len(countVec.get_feature_names()))

neg_doc_matrix = countVec_comment.transform(df[df['sentiment_one_hot'] == 0]['preprocessed_text'])
pos_doc_matrix = countVec_comment.transform(df[df['sentiment_one_hot'] == 1]['preprocessed_text'])
neg_tf_comment = np.sum(neg_doc_matrix, axis=0)
pos_tf_comment = np.sum(pos_doc_matrix, axis=0)
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
negative_term_freq_df_comment =term_freq_df_comment[term_freq_df_comment['negative_sent_score'] > 0.8].negative_sent_score
possitive_term_freq_df_comment =term_freq_df_comment[term_freq_df_comment['positive_sent_score'] > 0.8].positive_sent_score


print(negative_term_freq_df_comment)
joblib.dump(negative_term_freq_df_comment, "hate-words.pkl")
joblib.dump(possitive_term_freq_df_comment, "not-hate-words.pkl")




