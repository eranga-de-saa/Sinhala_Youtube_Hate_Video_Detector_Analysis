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

    countVec_comment = CountVectorizer()

    countVec_comment.fit(df['data'])

    negative_score_comment = train(traindf, countVec_comment)

    prediction = predict(testdf, negative_score_comment)

    print(classification_report(testdf['sentiment_one_hot'], prediction))

    pool.close()


def preprocess(item):
    comment = str(item["comment"])
    processed_comment = pr.process(comment)
    # print(processed_comment)
    if (processed_comment != "None") and (processed_comment is not None) and (processed_comment != ""):
        return processed_comment
    else:
        return None


def train(traindf, countVec_comment):

    neg_doc_matrix_comment = countVec_comment.transform(traindf[traindf['sentiment_one_hot'] == 0]['data'])
    pos_doc_matrix_comment = countVec_comment.transform(traindf[traindf['sentiment_one_hot'] == 1]['data'])
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
    # term_freq_df_comment = term_freq_df_comment.drop(term_freq_df_comment[(term_freq_df_comment['positive_sent_score'] < 0.6) & (term_freq_df_comment['positive_sent_score'] > 0.4)].index)
    # print(term_freq_df[term_freq_df['polarity_score'] < -0.5].sort_values(by='polarity_score', ascending=False).iloc[:15])
    # print(term_freq_df.columns)
    # print(term_freq_df)
    # positive_score_comment = term_freq_df_comment.positive_sent_score
    negative_score_comment = term_freq_df_comment.negative_sent_score
    # print(term_freq_df['total']['යන'])
    # print(positive_score_comment)
    # val_probability = []
    # print(positive_score["යන"])

    #     # for w in doc.split():
    #     #     print (w)


    # joblib.dump(positive_score_comment, "positive_score_comment_pickle.pkl")
    return negative_score_comment

def predict(testdf,positive_score_comment):
    testdf = testdf.reset_index()
    print(testdf)

    prediction = []
    for index, row in testdf.iterrows():
        print(str(index))
        # comment automatic annotation
        print(testdf['data'].iloc[index])
        sentiment_score_comment = [positive_score_comment[w] for w in testdf['data'].iloc[index].split() if w in positive_score_comment.index]

        if len(sentiment_score_comment) > 0:
            prob_score_comment = np.mean(sentiment_score_comment)
        else:
            prob_score_comment = np.random.random()

        prediction_comment = 1 if prob_score_comment < 0.5 else 0

        prediction.append(prediction_comment)
    return prediction


if __name__ == '__main__':
    main()
