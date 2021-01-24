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
from imblearn.over_sampling import SMOTE
import sys
sys.path.insert(0, '../')
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocessor as pr
from keras.utils import np_utils
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


comment_sentiModel = joblib.load("../../CommentSentimentAnalysis/MachineLearningApproach/comment_Lr.pkl")
comment_feature_mapper = joblib.load("../../CommentSentimentAnalysis/MachineLearningApproach/comment_feature_mapper.pkl")

political_words =  pd.read_excel (r'C:\Users\Eranga.95\Desktop\FYProject\PVocabulary.xlsx')

relious_ethnic_words =  pd.read_excel (r'C:\Users\Eranga.95\Desktop\FYProject\REVocabulary.xlsx')

sex_gender_words =  pd.read_excel (r'C:\Users\Eranga.95\Desktop\FYProject\SGVocabulary.xlsx')

political_vocabulary = [word.strip() for word in political_words['words']]

relious_ethnic_vocabulary = [word.strip() for word in relious_ethnic_words['words']]

sex_gender_vocabulary = [word.strip() for word in sex_gender_words['words']]

df_thumbnail = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/ThumnailTexts.xlsx')

def main():
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())  # for parallel processing
    df = pd.DataFrame(columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'posToNegCommentRatio', 'sentiment', 'category','pcount', 'recount', 'sgcount'])
    # df = pd.DataFrame(
    #     columns=['comment', 'otherMetadata', 'likeDislikeRatio', 'posToNegCommentRatio', 'sentiment', 'category'])

    i =0
    for x in range(1000):
        with open("../../output/"+str(x)+".json", encoding='utf-8') as f:
            pcount = 0
            recount = 0
            sgcount = 0
            wordcount = 0
            data = json.load(f)
            # global commentsText
            commentsText = ""
            # tags = data[0]["tags"]
            sentiment = data[0]['sentiment']
            category = data[0]['category']
            print(category)
            thumbnail_text = df_thumbnail['Thumbnail'].iloc[i]
            if 'tags' in data[0].keys():
                tags = str(' '.join(data[0]["tags"]))
                # print("tags" + tags)
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + " " + tags + ' ' + thumbnail_text)
            else :
                otherMetaData = pr.process(data[0]["title"] + " " + data[0]["description"] + ' ' + thumbnail_text)

            likes = int(data[0]["likeCount"])
            dislikes = int(data[0]["dislikeCount"])
            likeDislikeRatio = str(float(likes / dislikes))
            results = pool.map(preprocess, [item for item in data[0]["comments"]])
            postiveCount = 1
            negativeCount = 1
            for result in results:
                if result[0] is not None:
                    commentsText = commentsText + " " + result[0]
                    if result[1] == 1:
                        postiveCount = postiveCount + 1
                    else:
                        negativeCount = negativeCount + 1

                if result[2] is not None:
                     pcount = pcount+int(result[2])

                if result[3] is not None:
                     recount = recount+int(result[3])

                if result[4] is not None:
                    sgcount = sgcount +  int(result[4])

                if result[5] is not None:
                     wordcount = wordcount+ int(result[5])

            p_count = str(float((pcount * 100 / wordcount)))
            re_count = str(float((recount * 100 / wordcount)))
            sg_count = str(float((sgcount * 100 / wordcount)))

            posToNegCommentRatio = str(float(postiveCount / negativeCount))
            df.loc[i] = [commentsText] + [otherMetaData] + [likeDislikeRatio] + [posToNegCommentRatio] + [sentiment] + [category] + [p_count] + [re_count] + [sg_count]
            print(str(i)+" : "+df['posToNegCommentRatio'].loc[i])

        i += 1


    df['sentiment_one_hot'] = df['sentiment'].apply(lambda x: 0 if x == 'N' else 1)

    df['data'] = df['comment'] + " " + df['otherMetadata']


    mapper = DataFrameMapper([
        ('data',TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        (['pcount'], StandardScaler()),
        (['recount'], StandardScaler()),
        (['sgcount'], StandardScaler()),
    ])

    mapper.fit(df)
    data = mapper.transform(df)
    # joblib.dump(mapper, "Domain_feature_mapper.pkl")

    # Synthetic Minority Over-sampling Technique
    smote = SMOTE('minority')
    X_sm, Y_sm = smote.fit_sample(data, df['category'])

    encoder = LabelEncoder()
    encoder.fit(Y_sm)
    encoded_Y = encoder.transform(Y_sm)

    joblib.dump(encoder, "Domain_label_encoder.pkl")
    print(encoded_Y)

    dummy_y = np_utils.to_categorical(encoded_Y)
    print(dummy_y)
    X_train, X_test, y_train, y_test = train_test_split(X_sm, dummy_y, test_size=0.2)

    # X_train, X_test, y_train, y_test = train_test_split(data, df['category'], test_size=0.2)

    pool.close()

    model_DNN = Build_Model_DNN_Text(X_train.shape[1], 4)
    history_dropout = model_DNN.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=32)

    predicted = model_DNN.predict(X_test)
    y_pred_vector = np.argmax(predicted, axis=1)
    y_test_vector = np.argmax(y_test, axis=1)
    print(classification_report(y_test_vector, y_pred_vector))


    loss = history_dropout.history['loss']
    val_loss = history_dropout.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history_dropout.history['acc']
    val_acc = history_dropout.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model_DNN.save("Domain_ann.pkl")

    # joblib.dump(model_DNN, "Domain_ann.pkl")

    # print("oversamapled")
    # print("logistic Regression")
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    #
    # y_pred = clf.predict(X_test)
    #
    # print('oversampling accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred, target_names=df['category'].unique()))

    # print("SVM")
    # SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # SVM.fit(X_train, y_train)
    # # predict the labels on validation dataset
    # predictions_SVM = SVM.predict(X_test)
    # print(classification_report(y_test, predictions_SVM, target_names=df['category'].unique()))
    #
    # print("Random Forest")
    # clf = RandomForestClassifier(n_estimators = 300, criterion = "entropy", random_state = 0)
    # clf.fit(X_train, y_train)
    # predicted = clf.predict(X_test)
    # print(classification_report(y_test, predicted))

    # mapperCounter.fit(df)
    # dataCounter = mapper.transform(df)

    # Synthetic Minority Over-sampling Technique increased accuracy by 3%
    # smote = SMOTE('minority')

    # X_sm_counter, Y_sm_counter = smote.fit_sample(dataCounter, df['category'])

    # X_train_counter, X_test_counter, y_train_counter, y_test_counter = train_test_split(dataCounter,  df['category'],
    #                                                                                     test_size=0.2)
    # print("logistic Regression")
    # clf = LogisticRegression()
    # clf.fit(X_train_counter, y_train_counter)
    #
    # y_pred_counter = clf.predict(X_test_counter)
    #
    # print('oversampling accuracy %s' % accuracy_score(y_pred_counter, y_test_counter))
    # print(classification_report(y_test_counter, y_pred_counter, target_names=df['category'].unique()))

    # encoder = LabelEncoder()
    # encoder.fit(Y_sm)
    # encoded_Y = encoder.transform(Y_sm)
    #
    # joblib.dump(encoder, "Domain_label_encoder.pkl")
    # print(encoded_Y)
    # # convert integers to dummy variables (i.e. one hot encoded)
    # dummy_y = np_utils.to_categorical(encoded_Y)
    # print(dummy_y)



def preprocess(item):
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    if (processed_comment != "None") and (processed_comment is not None) and (processed_comment != ""):
        transformed = comment_feature_mapper.transform([processed_comment])
        sentiment = int(comment_sentiModel.predict(transformed))

        PCount = 0
        RECount = 0
        SGCount = 0
        wordCount=0

        for w in processed_comment.split():
            wordCount = wordCount + 1
            word = w
            if "/" in word:
                word = w[:w.index('/')]
            if word in political_vocabulary:
                PCount = PCount + 1
            elif word in relious_ethnic_vocabulary:
                RECount = RECount + 1
            elif word in sex_gender_vocabulary:
                SGCount = SGCount + 1

        # positive_count = positive_count / word_count
        # negative_count = negative_count / word_count

        # print(PCount, RECount, SGCount, wordCount)
        return processed_comment, sentiment, PCount, RECount, SGCount, wordCount
        # return processed_comment, sentiment
    else:
        return None,None,None,None,None,None
        # return None, None


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
    model.compile(loss='categorical_crossentropy',  # sparse_categorical_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    main()
