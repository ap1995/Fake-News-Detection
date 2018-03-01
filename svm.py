#!/usr/bin/env python

import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC

def run(train_file, valid_file, test_file, output_file):
    with open(train_file, 'rt') as f:
        reader = csv.reader(f, delimiter = '\t')
        data = list(reader)

    df = []
    y=[]
    for i in data:
        y.append(i[0])
    for i in data:
        df+=[' '.join(i)]

    vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english')
    X =vectorizer.fit_transform(df)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)

    clf = LinearSVC(random_state=0)
    clf.fit(X_train_tfidf, y)

    #### TEST DATA ###

    with open(test_file, 'rt') as f:
        reader = csv.reader(f, delimiter = '\t')
        test_data = list(reader)

    test =[]
    for i in test_data:
        test += [' '.join(i)]

    text_file = open(output_file, "w")
    for t in test:
        test_X = vectorizer.transform([t])
        X_test_tfidf = tfidf_transformer.transform(test_X)
        predicted = clf.predict(X_test_tfidf)
        text_file.write(predicted[0]+'\n')

train_file = 'train.tsv'
valid_file = 'valid.tsv'
test_file = 'test.tsv'
output_file = 'predictions.txt'
run(train_file, valid_file, test_file, output_file)
