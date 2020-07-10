#Credits: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import warnings
import pickle
import os
import random
import re
import json
import sys, getopt
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


def naive_bayes_clf(X_train, X_test, alpha, use_grid_search = False, grid_values = {}):
    if not use_grid_search:
        naive_clf = naive_bayes.MultinomialNB(alpha=alpha)
        naive_clf.fit(X_train,y_train)# predict the labels on validation dataset
        predictions_NB = naive_clf.predict(X_test)# Use accuracy_score function to get the accuracy
        save_model("naive_bayes_2_classes",naive_clf)
        print_metrics("Naive Bayes Classifier", y_test, predictions_NB)
    else:
        if not grid_values:
            print("Provide grid inputs")
            return
        else:
            naive_clf = naive_bayes.MultinomialNB()
            grid_clf = GridSearchCV(naive_clf, param_grid=grid_values)
            grid_clf.fit(X_train, y_train)
            predictions_NB = grid_clf.predict(X_test)
            print_metrics("Naive Bayes Classifier (Grid search)", y_test, predictions_NB)
            logging.info("Grid NB Best Parameters: {}".format(grid_clf.best_params_))
            logging.info("\n"+ pd.DataFrame.from_dict(grid_clf.cv_results_)[["params", "mean_test_score"]].to_csv(sep=' ', index=False))
            save_model("naive_bayes_2_class_250", grid_clf)
def svm_clf(X_train, X_test, C, gamma, use_grid_search = False, grid_values = {}):
    if not use_grid_search:
        svm_clf = svm.SVC(C=C, gamma=gamma)
        svm_clf.fit(X_train, y_train)
        predictions_SVM = svm_clf.predict(X_test)# Use accuracy_score function to get the accuracy
        save_model("svm_2_classes", svm_clf)
        print_metrics("SVM Classifier", y_test, predictions_SVM)
    
    else:
        if not grid_values:
            print("Provide grid inputs")
            return
        else:
            svm_clf = svm.SVC()
            grid_clf = GridSearchCV(svm_clf, param_grid=grid_values)
            grid_clf.fit(X_train, y_train)
            predictions_SVM = grid_clf.predict(X_test)
            print_metrics("SVM Classifier (Grid search)", y_test, predictions_SVM)
            logging.info("Grid SVM Best Parameters: {}".format(grid_clf.best_params_))
            logging.info("\n"+ pd.DataFrame.from_dict(grid_clf.cv_results_)[["params", "mean_test_score"]].to_csv(sep=' ', index=False))
def get_tfidf_vectorizer(corpus, max_features=500):
    logging.info("Data vectorized using Tfidf Vectorizer")
    Tfidf_vect = TfidfVectorizer(max_features=max_features)
    Tfidf_vect.fit(corpus)
    return Tfidf_vect

def get_count_vectorizer(corpus, n_grams, max_features = 500):
    logging.info("Data vectorized using Count Vectorizer with {} ngrams".format(n_grams))
    Count_vect = CountVectorizer(ngram_range=(n_grams, n_grams), max_features = max_features)
    Count_vect.fit(corpus)
    return Count_vect


def print_metrics(clf_name, y_test, predicted):
    logging.info("classifier: {}".format(clf_name))
    logging.info("\n" + classification_report(y_test, predicted))

def save_model(clf_name, clf):
    with open(os.path.join(dir_path,"classifiers/{}.pkl".format(clf_name)), "wb") as clf_f:
        pickle.dump(clf, clf_f)

def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    logging.info("File loaded: {}".format(file_path))
    return data



if __name__ == "__main__":
    dir_path = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP/bbc"
    CUSTOM_SEED=42
    np.random.seed(CUSTOM_SEED)
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:x:y:c:v:n:",["model=","X=", "y=", "corpus=", "vectorizer=","ngrams="])
    except getopt.GetoptError:
        print('python3 {} -m <nb|svm> -x <X data file> -y <y data file> -c <corpus file> -v <tfidf|ngrams> -n <1|2>'.format(sys.argv[0]))
        sys.exit(2)
    ngrams = None
    for opt, arg in opts:
        if opt == '-h':
            print('python3 {} -m <nb|svm> -x <X data file> -y <y data file> -c <corpus file> -v <tfidf|ngrams> -n <1|2>'.format(sys.argv[0]))
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-x", "--X"):
            X_path = "data/" + arg
        elif opt in ("-y", "--y"):
            y_path = "data/" + arg
        elif opt in ("-c", "--corpus"):
            corpus_path = "data/" + arg
        elif opt in ("-v", "--vectorizer"):
            vectorizer_selection = arg
        elif opt in ("-n", "--ngrams"):
            ngrams = int(arg)
        
    if ngrams == None: ngrams = 1


    #all data that i am using has been segmented, stop words and no burmese text removed.


    if vectorizer_selection == "ngrams":
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(dir_path,"code/logs/{}_{}_{}_{}".format(model, vectorizer_selection, str(ngrams), str(X_path[5:-5]))), filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")
        print("logs/{}_{}_{}_{}".format(model, vectorizer_selection, str(ngrams), str(X_path[5:-5])))
    elif vectorizer_selection == "tfidf":
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(dir_path,"code/logs/{}_{}_{}".format(model, vectorizer_selection, str(X_path[5:-5]))), filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")
        print("logs/{}_{}_{}".format(model, vectorizer_selection, str(X_path[5:-5])))
    logging.info("hello")    
    corpus = load_json_file(os.path.join(dir_path, corpus_path))
    X = load_json_file(os.path.join(dir_path, X_path))
    y = load_json_file(os.path.join(dir_path, y_path))
    

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20)



    # logging.info("Corpus: {}".format(corpus_path))
    # logging.info("X data : {}".format(X_path))
    # logging.info("y data: {}".format(y_path))
    logging.info("Data loaded")
    logging.info("Num articles: {}".format(len(X)))

    logging.info("Training data size : {}".format(len(y_train)))
    logging.info("Testing data size: {}".format(len(y_test)))

    #This is not one-hot encoding but it replaces the name of the class with a number. So the number of columns will still be 1.

    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)
    logging.info("Y encoded")
    

    if vectorizer_selection == "tfidf":
        vectorizer = get_tfidf_vectorizer(corpus = corpus, max_features = 5000)
    elif vectorizer_selection == "ngrams":
        if ngrams != None:
            vectorizer = get_count_vectorizer(corpus = corpus, n_grams = ngrams, max_features = 5000)
        else:
            vectorizer = get_count_vectorizer(corpus = corpus, n_grams = 1, max_features = 5000)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    

    logging.info("Feature names : {} ...".format(vectorizer.get_feature_names()[:20]))
    logging.info("Vectorized X train dimensions: {}".format(np.shape(X_train_vectorized)))
    logging.info("Vectorized X test dimensions: {}".format(np.shape(X_test_vectorized)))



    # Different values were tested using GridSearch and the following values yielded the best results.

    svm_grid_values = {
            "C" : (0.01, 0.1, 1, 10, 100),
            "gamma":(10, 1, 0.1, 0.01),
            "kernel": ("linear", "rbf")
            }

    nb_grid_values = {
            "alpha":(0.001, 0.01, 0.1, 0.2, 0.3,0.4)
            }

    if model == "svm":
        svm_clf(X_train_vectorized, X_test_vectorized, 1,2, True, svm_grid_values)
    elif model == "nb":
        naive_bayes_clf(X_train_vectorized, X_test_vectorized, 0.2, True, nb_grid_values)
