from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import multiprocessing
from sklearn import utils
import numpy as np
import sys
sys.path.insert(0, '../../')
from preprocess import preprocessor as pr
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

df = pd.read_excel (r'C:\Users\Eranga.95\Desktop\FYProject/FYP Comments.xlsx')

df['preprocessed_text'] = df.apply(lambda x: preprocess(x["Comment"]), axis=1)

df['sentiment_one_hot'] = df['Sentiment'].apply(lambda x: 0 if x == 'H' else 1)


def labelize_comments_ug(comments, label):
    result = []
    prefix = label
    for i, t in zip(comments.index, comments):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


all_x = df['preprocessed_text']  # pd.concat([x_train, x_validation, x_test])
all_x_w2v = labelize_comments_ug(all_x, 'all')
cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2,
                         workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                        total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2,
                       workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                      total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha


def get_w2v_mean(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word], model_ug_sg[word]).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_w2v_sum(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word], model_ug_sg[word]).reshape((1, size))
        except KeyError:
            continue
    return vec

x_train, x_test, y_train, y_test = train_test_split(df['preprocessed_text'], df['sentiment_one_hot'], test_size=0.3)


train_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_test]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_mean, y_train)
print(clf.score(validation_vecs_cbowsg_mean, y_test))
print(classification_report(y_test, clf.predict(validation_vecs_cbowsg_mean)))

train_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_train]))
validation_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_test]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_sum, y_train)
print(clf.score(validation_vecs_cbowsg_sum, y_test))


