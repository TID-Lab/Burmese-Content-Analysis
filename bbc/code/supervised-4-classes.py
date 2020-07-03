#Credits: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import warnings
import pickle
import os
import random
import re
import json
warnings.simplefilter(action='ignore', category=FutureWarning)
'''
bbc.json file contains segmented data with stop words removed and non Burmese text removed.
'''



def transform_to_dataset(tagged_articles):
    """
    Create X and y aray with article text and article category
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return: 
    """
    X, y = [], [] 
    for article in tagged_articles:
        X.append(article["text"])
        y.append(article["category"])
    return X, y


def naive_bayes_clf(alpha):
    naive_clf = naive_bayes.MultinomialNB(alpha=alpha)
    naive_clf.fit(X_train_Tfidf,y_train)# predict the labels on validation dataset
    predictions_NB = naive_clf.predict(X_test_Tfidf)# Use accuracy_score function to get the accuracy
    save_model("naive_bayes",naive_clf)
    print_metrics("Naive Bayes Classifier", y_test, predictions_NB)


def svm_clf(C, gamma):
    svm_clf = svm.SVC(C=C, gamma=gamma)
    svm_clf.fit(X_train_Tfidf, y_train)
    predictions_SVM = svm_clf.predict(X_test_Tfidf)# Use accuracy_score function to get the accuracy
    save_model("svm", svm_clf)
    print_metrics("SVM Classifier", y_test, predictions_SVM)

def print_metrics(clf_name, y_test, predicted):
    print("classifier: {}".format(clf_name))
    print(classification_report(y_test, predicted))

def save_model(clf_name, clf):
    with open("classifiers/{}.pkl".format(clf_name), "wb") as clf_f:
        pickle.dump(clf, clf_f)


CUSTOM_SEED=42
np.random.seed(CUSTOM_SEED)

dir_path = "/home/harshil/Harshil/gt/spring2020/research2/ml-evaluation-models/bbc"

data_path = "data/bbc.json"


corpus_path = "data/corpus.json"  #this file contains all words in all articles in a single array after removal of stop words and non burmese words. A list of documents.

with open(os.path.join(dir_path, data_path), "r") as data_f:
    data = json.load(data_f)
    random.shuffle(data)

with open(os.path.join(dir_path, corpus_path), "r") as corpus_f:
    corpus = json.load(corpus_f)


print("Dataset Information")
print("-"*50)
print("Num articles: {}".format(len(data)))

train_test_cutoff = int(.80 * len(data)) 
training_articles = data[:train_test_cutoff]
testing_articles = data[train_test_cutoff:]
train_val_cutoff = int(.25 * len(training_articles))
validation_articles = training_articles[:train_val_cutoff]
training_articles = training_articles[train_val_cutoff:]


#After transforming to dataset, a huge matrix is created with each row representing a data point in X train and the columns representing the features. 
X_train, y_train = transform_to_dataset(training_articles)
X_test, y_test = transform_to_dataset(testing_articles)
X_val, y_val = transform_to_dataset(validation_articles)

#This is not one-hot encoding but it replaces the name of the class with a number. So the number of columns will still be 1.

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)
y_val = Encoder.fit_transform(y_val)


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus)
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)


# Different values were tested using GridSearch and the following values yielded the best results.

naive_bayes_clf(0.2)
svm_clf(1,2)

