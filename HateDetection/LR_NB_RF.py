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
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, '../')
from preprocess import preprocessor as pr

comment_sentiModel = joblib.load("../CommentSenti/comment_Lr.pkl")
comment_feature_mapper = joblib.load("../CommentSenti/comment_feature_mapper.pkl")

negation_words = {"නැ", "නෑ", "නැහැ", "නැත", "නැති", "බැ", "බෑ", "බැහැ", "බැරිය", "එපා"}


df_positive_words = pd.read_excel(
    r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\positivewords.xlsx')

df_negative_words = pd.read_excel(
    r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\HateWords.xlsx')
df_thumbnail = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/ThumnailTexts.xlsx')


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
    df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'posToNegCommentRatio', 'sentiment'])

    i = 0
    for x in range(1000):
        with open("../output/" + str(x) + ".json", encoding='utf-8') as f:
            data = json.load(f)
            # global commentsText
            commentsText = ""
            # tags = data[0]["tags"]
            sentiment = data[0]['sentiment']
            thumbnail_text = df_thumbnail['Thumbnail'].iloc[i]
            if 'tags' in data[0].keys():
                tags = str(' '.join(data[0]["tags"]))
                # print("tags" + tags)
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + " " + tags + ' ' + thumbnail_text)
            else:
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + ' ' + thumbnail_text)

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
            print(str(i) + " : " + df['posToNegCommentRatio'].loc[i])

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
    pool.close()
    df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)
    df['sentiment_one_level_hate'] = df['sentiment'].apply(lambda x: 1 if x == 'N' else 0)

    # mapper = DataFrameMapper([
    #     (['posToNegCommentRatio'], StandardScaler()),
    #     ('otherMetadata', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
    #     (['likeDislikeRatio'], StandardScaler()),
    # ])

    mapper = DataFrameMapper([
        (['posToNegCommentRatio'], StandardScaler()),
        ('otherMetadata', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        (['likeDislikeRatio'], StandardScaler()),
    ])


    mapper.fit(df)
    label = df['sentiment_one_hot']

    features = mapper.transform(df)

    # x, x_test, y, y_test = train_test_split(features, label, test_size=0.2, train_size=0.8, random_state=0)
    print("logistic regression")
    clf = LogisticRegression()
    clf.fit(features, label)

    # predicted = clf.predict(x_test)

    # print(classification_report(y_test, predicted))


    # x1, x_test1, y1, y_test1 = train_test_split(features, df['sentiment_one_level_hate'], test_size=0.2, train_size=0.8, random_state=0)
    print("logistic regression")
    clf_level = LogisticRegression()
    clf_level.fit(features, df['sentiment_one_level_hate'])

    # predicted1 = clf_level.predict(x_test1)

    # print(classification_report(y_test1, predicted1))


    # print("SVM")
    #
    # SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # SVM.fit(x_test, y)
    # # predict the labels on validation dataset
    # predictions_SVM = SVM.predict(x_test)
    # print(classification_report(y_test, predictions_SVM))
    # print("Random Forest")
    # clf = RandomForestClassifier(n_estimators = 300, criterion = "entropy", random_state = 0)
    # clf.fit(x, y)
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))

    joblib.dump(clf, "HateDetection_LR.pkl")
    joblib.dump(clf_level, "Level_of Hate_LR.pkl")
    joblib.dump(mapper, "HD_featureMapper_LR.pkl")

    # print(y_test)
    #
    # z = np.dot(clf.coef_, x_test.T) + clf.intercept_
    # hypo = 1/(1+np.exp(-z))
    # print("Level of hate =" + str(hypo))

    # featuresNB = df['comment'] + ' ' + df['otherMetadata']
    #
    # cvec = CountVectorizer(ngram_range=(1, 3), max_features=5000)
    #
    # cvec.fit(featuresNB)
    #
    # nb_x, nb_x_test, nb_y, nb_y_test = train_test_split(cvec.transform(featuresNB), label, test_size=0.2,
    #                                                     train_size=0.8,
    #                                                     random_state=0)
    # print("MNB")
    # nb = MultinomialNB()
    # nb.fit(nb_x, nb_y)
    # predicted_nb = nb.predict(nb_x_test)
    #
    # print(classification_report(nb_y_test, predicted_nb))
    #
    # print("ANN")
    #
    # model = Sequential()
    # model.add(Dense(64, activation='relu', input_dim=x.shape[1]))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    # # print(classification_report(y_test_vector, y_pred_vector))
    #
    # model.fit(x, y, validation_data=(x_test, y_test),
    #           epochs=20,
    #           batch_size=64,
    #           verbose=1)
    # ann_prediction=model.predict_classes(x_test)
    # print(classification_report(nb_y_test, ann_prediction))

    # print(predicted_nb)
    #
    # somelist = nb.predict_proba(nb_x_test)
    #
    # for obj in somelist:
    #     print(obj)
    #     hate = float(obj[0] / obj[0] + obj[1])
    #     print(hate)

    # print(nb.predict_proba(nb_x_test))
    # print(nb.predict_log_proba(nb_x_test))
    #
    # joblib.dump(nb, "HateDetection_Nb.pkl")
    # joblib.dump(cvec, "HD_featureMapper_NB.pkl")




def preprocess(item):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None) and (processed_comment.strip() != ''):

        positive_count = 0
        negative_count = 0
        word_count = 0
        comment_neg = processed_comment
        for i, w in enumerate(processed_comment.split()):
            word_count = word_count + 1
            word = w
            if "/" in word:
                word = w[:w.index('/')]
            if word in positive_vocabulary:
                positive_count = positive_count + 1
            elif word in negative_vocabulary:
                negative_count = negative_count + 1
            elif word in negation_words:
                # print("comment :" + comment)
                word = processed_comment.split()[i]
                # print("Negation word :" + word)
                previous_word = word = processed_comment.split()[i - 1]
                new_previous_word = "not_" + previous_word
                # print("Previous Word : " + previous_word)
                before_previous_word = processed_comment[:processed_comment.find(previous_word)]
                # print("Before Previous Word: " + before_previous_word)
                after_previous_word = processed_comment[processed_comment.find(previous_word) + len(previous_word):]
                # print("After previous Word:" + after_previous_word)
                comment_neg = before_previous_word + ' ' + new_previous_word + ' ' + after_previous_word

        positive_count = positive_count / word_count
        negative_count = negative_count / word_count

        # data = processed_comment, [positive_count], [negative_count]

        data = {'preprocessed_text': [comment_neg], 'positive_count': [positive_count],
                'Negative_count': [negative_count]}

        comment_df = pd.DataFrame(data)
        # print(comment_df)
        # comment_df = pd.DataFrame(data, columns=['preprocessed_text', 'positive_count', 'Negative_count'])

        transformed = comment_feature_mapper.transform(comment_df)
        sentiment = int(comment_sentiModel.predict(transformed))
        # print('sentiment: ')
        # print(sentiment)
        # commentsText += processed_comment
        return processed_comment, sentiment
    else:
        return None, None



if __name__ == '__main__':
    main()

# level of hate
# tranformed = mapper.transform(df)
# print(tranformed[0][2])
# prediction = logisticRegression.predict(tranformed)
# print(prediction)
#
# clf = logisticRegression
# z = np.dot(clf.coef_, tranformed.T) + clf.intercept_
# hypo = 1/(1+np.exp(-z))
# print("Level of hate =" + str(hypo))