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


df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/FYP Comments.xlsx')
# print (df)

# print(df.iloc[298])

df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)
print(df.iloc[867])
x_train, x_test, y_train, y_test = train_test_split(df['Comment'], df['sentiment_one_hot'], test_size=0.3)



def performace_evaluation(pipeline, x_train, y_train, x_test, y_test):
    start_time = time()
    model_fit = pipeline.fit(x_train,y_train)
    modal_prediction = model_fit.predict(x_test)
    train_time = time() - start_time
    accuracy = accuracy_score(y_test, modal_prediction)
    # print(classification_report(y_test, modal_prediction))
    return accuracy, train_time


countVec = CountVectorizer()
tfidfvec = TfidfVectorizer()
lr = LogisticRegression()
n_features = np.arange(400, 2001, 100)


def feature_checker(vectorizer=countVec, nfeatures=n_features, ngram_range=(1, 1), classifier=lr):
    result = []
    # print(classifier)
    # print("\n")
    for n in nfeatures:
        vectorizer.set_params(max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Testing with {} features".format(n))
        feature_accuracy, tt_time = performace_evaluation(checker_pipeline, x_train, y_train, x_test, y_test)
        result.append((n, feature_accuracy, tt_time))
    return result

print("\nResults for counter unigrams")
feature_result_ug = feature_checker()

print("\nResults for counter bigram")
feature_result_bg = feature_checker(ngram_range=(1, 2))

print("\nResults for counter trigram")
feature_result_tg = feature_checker(ngram_range=(1, 3))




print("\nResults for tfidf unigrams")
feature_result_ugt = feature_checker(vectorizer=tfidfvec)

print("\nResults for tfidf bigram")
feature_result_bgt = feature_checker(ngram_range=(1, 2), vectorizer=tfidfvec)

print("\nResults for tfidf trigram")
feature_result_tgt = feature_checker(ngram_range=(1, 3), vectorizer=tfidfvec)



plot_unigram = pd.DataFrame(feature_result_ug, columns=['nfeatures', 'accuracy', 'train_time'])

plot_bigram = pd.DataFrame(feature_result_bg, columns=['nfeatures', 'accuracy', 'train_time'])

plot_trigram = pd.DataFrame(feature_result_tg, columns=['nfeatures', 'accuracy', 'train_time'])

plot_unigramt = pd.DataFrame(feature_result_ugt, columns=['nfeatures', 'accuracy', 'train_time'])

plot_bigramt = pd.DataFrame(feature_result_bgt, columns=['nfeatures', 'accuracy', 'train_time'])

plot_trigramt = pd.DataFrame(feature_result_tgt, columns=['nfeatures', 'accuracy', 'train_time'])


plt.figure(figsize=(8, 6))
plt.plot(plot_unigram["nfeatures"], plot_unigram["accuracy"], label="counter unigram")
plt.plot(plot_bigram["nfeatures"], plot_bigram["accuracy"], label="counter bigram")
plt.plot(plot_trigram["nfeatures"], plot_trigram["accuracy"], label="counter trigram")
plt.plot(plot_unigramt["nfeatures"], plot_unigramt["accuracy"], label="tfidf unigram")
plt.plot(plot_bigramt["nfeatures"], plot_bigramt["accuracy"], label="tfidf bigram")
plt.plot(plot_trigramt["nfeatures"], plot_trigramt["accuracy"], label="tfidf trigram")
plt.title("n_gram accuracy ")
plt.xlabel("Number of features")
plt.ylabel("Test accuracy")
plt.legend()
plt.show()
