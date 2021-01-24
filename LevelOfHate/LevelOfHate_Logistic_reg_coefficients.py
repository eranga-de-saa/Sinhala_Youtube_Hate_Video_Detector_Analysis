# create a comment sentiment classifier
# get the videos overall positive/negative comment ratio
# train logistic regression model for overall positive/negative comment ratio and like/dislike ratio
# extract the coefficients ... create a simple equation from the coefficient
# Use that equation to measure the level of hate
#
# second method using lexicle analysis get average positivity for each feature-> make a model to get the coefficients
# use them to calculate the level of hate
#
#


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

dataset = defaultdict(list)

# create a dataframe to store dataset
df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'sentiment'])


i =0
for x in range(604):
    with open("../output/"+str(x)+".json", encoding='utf-8') as f:
        data = json.load(f)
    commentsText = ""
    replyText = ""
    replies = defaultdict(list)
# store sentiment of the data
    sentiment = data[0]['sentiment']
    otherMetaData = data[0]["title"] + data[0]["description"]  # + item["tags"]
    likes = int(data[0]["likeCount"])
    dislikes = int(data[0]["dislikeCount"])
    likeDislikeRatio = str(float(likes/dislikes))
    for item in data[0]["comments"]:
        commentsText += item["comment"]
        if 'replies' in item.keys():
            for reply in item['replies']:
                replyText += reply['replyComment']
    df.loc[i] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [sentiment]
    i += 1
# class variable binerization
# change binerization to classify hate as 1
df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 1 if x == 'N' else 0)

# Comment lexical analysis
# print(df.head())

countVec_comment = CountVectorizer()
countVec_comment.fit(df['comment'])

# print(len(countVec.get_feature_names()))

neg_doc_matrix_comment = countVec_comment.transform(df[df['sentiment_one_hot'] == 0]['comment'])
pos_doc_matrix_comment = countVec_comment.transform(df[df['sentiment_one_hot'] == 1]['comment'])
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

# val_probability = []
# print(positive_score["යන"])

#     # for w in doc.split():
#     #     print (w)
# Othermeta data analysis

countVec_other = CountVectorizer()
countVec_other.fit(df['otherMetadata'])

# print(len(countVec.get_feature_names()))

neg_doc_matrix_other = countVec_other.transform(df[df['sentiment_one_hot'] == 0]['otherMetadata'])
pos_doc_matrix_other = countVec_other.transform(df[df['sentiment_one_hot'] == 1]['otherMetadata'])
neg_tf_other = np.sum(neg_doc_matrix_other, axis=0)
pos_tf_other = np.sum(pos_doc_matrix_other, axis=0)
neg_other = np.squeeze(np.asarray(neg_tf_other))
pos_other = np.squeeze(np.asarray(pos_tf_other))
term_freq_df_other = pd.DataFrame([neg_other, pos_other], columns=countVec_other.get_feature_names()).transpose()
term_freq_df_other.columns = ['negative', 'positive']
term_freq_df_other['total'] = term_freq_df_other['negative'] + term_freq_df_other['positive']
term_freq_df_other['positive_sent_score'] = term_freq_df_other['positive'] / term_freq_df_other['total']
term_freq_df_other['negative_sent_score'] = term_freq_df_other['negative'] / term_freq_df_other['total']
term_freq_df_other['polarity_score'] = 2*term_freq_df_other['positive_sent_score'] - 1
term_freq_df_other = term_freq_df_other.drop(term_freq_df_other[(term_freq_df_other['positive_sent_score'] < 0.6) & (term_freq_df_other['positive_sent_score'] > 0.4)].index)
# print(term_freq_df[term_freq_df['polarity_score'] < -0.5].sort_values(by='polarity_score', ascending=False).iloc[:15])
# print(term_freq_df.columns)
# print(term_freq_df)
positive_score_other = term_freq_df_other.positive_sent_score

# save in pickle format
joblib.dump(positive_score_comment, "positive_score_comment_pickle.pkl")
joblib.dump(positive_score_other, "positive_score_other_pickle.pkl")


classification_df = pd.DataFrame(columns=['comment_sentiment', 'otherMetadata_sentiment', 'likeDislikeRatio', 'sentiment'])
classification_df = classification_df.astype({"comment_sentiment": float,  "otherMetadata_sentiment": float, "likeDislikeRatio": float, "sentiment": bool})

for index, row in df.iterrows():

    # comment automatic annotation
    sentiment_score_comment = [positive_score_comment[w] for w in df['comment'].iloc[index].split() if w in positive_score_comment.index]

    if len(sentiment_score_comment) > 0:
        prob_score_comment = np.mean(sentiment_score_comment)
    else:
        prob_score_comment = np.random.random()

    prediction_comment = prob_score_comment


    #other meta data automatic annotaion
    sentiment_score_other = [positive_score_other[w] for w in df['otherMetadata'].iloc[index].split() if
                               w in positive_score_other.index]
    if len(sentiment_score_other) > 0:
        prob_score_other = np.mean(sentiment_score_other)
    else:
        prob_score_other = np.random.random()

    prediction_other = prob_score_other

    likeDislikeRatio = df['likeDislikeRatio'].iloc[index]
    sentiment = df['sentiment_one_hot'].iloc[index]
    classification_df.loc[index] = [prediction_comment] + [prediction_other] + [likeDislikeRatio] + [sentiment]

print(classification_df[['comment_sentiment', 'otherMetadata_sentiment', 'likeDislikeRatio']].head())


# Train split
x_train, x_test, y_train, y_test = train_test_split(classification_df[['comment_sentiment', 'otherMetadata_sentiment', 'likeDislikeRatio']][:601], classification_df['sentiment'][:601], test_size=0.3)
#
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
#
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

print(len(classification_df))
print(logreg.coef_[0])
print(logreg.intercept_)

print("weights")
print("bias, comment, otherMeta , Like/dislike")
print(np.hstack((logreg.intercept_[:, None], logreg.coef_)))

i = 603
z = logreg.intercept_ + float(classification_df['comment_sentiment'].iloc[i]) * float(logreg.coef_[0][0]) + float(classification_df['otherMetadata_sentiment'].iloc[i]) * float(logreg.coef_[0][1]) + float(classification_df['likeDislikeRatio'].iloc[i]) * float(logreg.coef_[0][2])
hypo = 1/(1+np.exp(-z))
print("Level of hate =" + str(hypo))

# joblib.dump(logreg.py, "logistic_regression_sentiment_model.pkl")



# ensemble_classifier

# clf1 = LogisticRegression()
# clf2 = LinearSVC()
# clf3 = MultinomialNB()
# clf4 = RidgeClassifier()
# clf5 = PassiveAggressiveClassifier()
#
# ensemble_classifier = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('mnb', clf3), ('rcs', clf4), ('pac', clf5)], voting= 'hard')
# ensemble_classifier.fit(x_train, y_train)
# accuracy = ensemble_classifier.score(x_test, y_test)
# print('Accuracy of ensemble classifier on test set: {:.2f}'.format(accuracy))
#
# joblib.dump(ensemble_classifier, "ensemble_classifier_model.pkl")

# print(logreg.py.decision_function(classification_df.iloc[1]))

# prediction = [1 if t > 0.56 else 0 for t in val_probability]
# print(prediction)
# test_actual = y_test.tolist()
# print(test_actual)
# # from sklearn.metrics import accuracy_score
# print(accuracy_score(test_actual, prediction))

# for doc in x_test:
#     sentiment_score = [positive_score[w] for w in doc.split() if w in positive_score.index]
#     # print(hmean_scores)
#     if len(sentiment_score) > 0:
#         prob_score = np.mean(sentiment_score)
#     else:
#         prob_score = np.random.random()
#     val_probability.append(prob_score)
#     # print(val_probability)
#     classification_df = doc['comment']
#
# #
# prediction = [1 if t > 0.56 else 0 for t in val_probability]
# print(prediction)
# test_actual = y_test.tolist()
# print(test_actual)
# # from sklearn.metrics import accuracy_score
# print(accuracy_score(test_actual, prediction))