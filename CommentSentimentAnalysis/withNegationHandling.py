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
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
# from preprocess import preprocessor as pr
import multiprocessing as mp
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '../')
from preprocess import preprocessor as pr
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis/FYP Comments.xlsx')

df_positive_words =  pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\positivewords.xlsx')

df_negative_words=  pd.read_excel (r'C:\Users\Eranga.95\Desktop\python-youtube-api-master - JSON\youtube-analysis_V3\youtube-analysis\HateWords.xlsx')

negation_words = {"නැ", "නෑ", "නැහැ", "නැත", "නැති", "බැ", "බෑ", "බැහැ", "බැරිය", "එපා"}

# print (df)

# print(df_negative_words.head())
#
# print(df_positive_words.head())

# print(df_positive_words.keys())

positive_vocabulary = [word.strip() for word in df_positive_words['words']]
negative_vocabulary = [word.strip() for word in df_negative_words['words']]

# print(vocabulary)

# print(df.iloc[298])

def preprocess(item):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    comment = str(item)
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None):
        # commentsText += processed_comment
        return processed_comment
    else:
        return None

# for index, row in df.iterrows():
#
#     df['preprocessed_text'].loc[index] = preprocess(df['Comment'].iloc[index])

df['preprocessed_text'] = df.apply(lambda x: preprocess(x["Comment"]), axis=1)

df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)
# print(df['preprocessed_text'])
# x_train, x_test, y_train, y_test = train_test_split(df['Comment'], df['sentiment_one_hot'], test_size=0.3)

plt.figure(figsize=(10, 4))
df['sentiment_one_hot'].value_counts().plot(kind='bar')
plt.show()


classification_df = pd.DataFrame(columns=['preprocessed_text', 'positive_count', 'Negative_count', 'sentiment'])
classification_df = classification_df.astype({"preprocessed_text": str, "positive_count": float, "Negative_count": float, "sentiment": bool})

neg_count = 0

for index, row in df.iterrows():
    positive_count = 1
    negative_count = 1
    word_count = 1
    # for w in df['preprocessed_text'].iloc[index].split():
    for i, w in enumerate(df['preprocessed_text'].iloc[index].split()):
        word_count = word_count+1
        word = w
        if "/" in word:
            word = w[:w.index('/')]
        if word in positive_vocabulary:
            positive_count = positive_count + 1
        elif word in negative_vocabulary:
            negative_count = negative_count + 1
        elif word in negation_words:
            neg_count = neg_count + 1
            comment = df['preprocessed_text'].iloc[index]
            print("comment :" + comment)
            word = df['preprocessed_text'].iloc[index].split()[i]
            print("Negation word :" + word)
            previous_word = word = df['preprocessed_text'].iloc[index].split()[i - 1]
            new_previous_word = "not_" + previous_word
            print("Previous Word : " + previous_word)
            before_previous_word = comment[:comment.find(previous_word)]
            # print("Before Previous Word: " + before_previous_word)
            after_previous_word = comment[comment.find(previous_word) + len(previous_word):]
            # print("After previous Word:" + after_previous_word)
            df['preprocessed_text'].iloc[index] = before_previous_word + ' ' + new_previous_word + ' ' + after_previous_word
            print(df['preprocessed_text'].iloc[index])
        # print(w[:w.index('/')])
    # sentiment = df['sentiment_one_hot'].iloc[index]
    positive_count = positive_count / word_count
    negative_count = negative_count / word_count
    classification_df.loc[index] = [df['preprocessed_text'].iloc[index]] + [positive_count] + [negative_count] + [df['sentiment_one_hot'].iloc[index]]

print(neg_count)

mapper = DataFrameMapper([
    ('preprocessed_text', CountVectorizer(ngram_range=(1, 3), max_features=5000)),
    (['positive_count'], StandardScaler() ),
    (['Negative_count'], StandardScaler()),
])


features = mapper.fit_transform(classification_df)


joblib.dump(mapper, "comment_feature_mapper.pkl")

label = df['sentiment_one_hot']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, train_size=0.8, random_state = 0)

print("Logistic regression")
clf = LogisticRegression()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
print(classification_report(y_test, predicted))

joblib.dump(clf, "comment_Lr.pkl")

# SVM

# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(X_train, y_train)
# # predict the labels on validation dataset
# predictions_SVM = SVM.predict(X_test)
# print(classification_report(y_test, predictions_SVM))



# MNB

# clf = MultinomialNB()
# clf.fit(X_train, y_train)
#
# predicted = clf.predict(X_test)
# print(classification_report(y_test, predicted))

# Random Forest

# clf = RandomForestClassifier(n_estimators = 300, criterion = "entropy", random_state = 0)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print(classification_report(y_test, predicted))

# ANN

# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history_dropout = model.fit(X_train, y_train,
#                               validation_data=(X_test, y_test),
#                               epochs=50,
#                               batch_size=64,
#                               verbose=1)
#
# predicted = model.predict_classes(X_test)
# print(predicted)
# print(classification_report(y_test, predicted))
#
#
# # history_dropout = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), batch_size=32)
# # history_dropout = model.fit_generator(datagen.flow(x_train, y_train),
# #                                       steps_per_epoch=len(x_train) / 32,
# #                                       epochs=10,
# #                                       validation_data=(x_test, y_test),
# #                                       callbacks=callbacks_list)
#
# loss = history_dropout.history['loss']
# val_loss = history_dropout.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# acc = history_dropout.history['acc']
# val_acc = history_dropout.history['val_acc']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



# comment_feature_mapper = TfidfVectorizer(ngram_range=(1, 3))
# positive_feature_mapper = TfidfVectorizer()
# positive_feature_mapper.fit(df_positive_words)
#
# negative_feature_mapper = TfidfVectorizer()
# negative_feature_mapper.fit(df_negative_words)
#
#
# mapper = DataFrameMapper([
#     ('preprocessed_text', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
#     ('preprocessed_text', TfidfVectorizer(min_df=1, vocabulary=positive_vocabulary)),
#     ('preprocessed_text', TfidfVectorizer(min_df=1, vocabulary=negative_vocabulary)),
# ])
#
#
# features = mapper.fit_transform(df)
#
#
# joblib.dump(comment_feature_mapper, "comment_feature_mapper.pkl")
#
# label = df['sentiment_one_hot']
#
# X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, train_size=0.8, random_state = 0)
#
# clf = LogisticRegression()
# clf.fit(X_train, y_train)
#
# predicted = clf.predict(X_test)
#
#
# print("bias, comment, otherMeta , Like/dislike")
# print(np.hstack((clf.intercept_[:, None], clf.coef_)))
