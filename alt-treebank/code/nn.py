'''
Credits: https://becominghuman.ai/part-of-speech-tagging-tutorial-with-the-keras-deep-learning-library-d7f93fa05537
'''
import numpy as np
import nltk
import random
import os
import tensorflow as tf
import re
from ast import literal_eval
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import History 
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report

def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """    
    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'    # plot model loss
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('loss')
    ax1.tick_params('y')
    ax1.legend(loc='upper right', shadow=False)
    ax1.set_title('Model loss through #epochs', color=orange, fontweight='bold')    # plot model accuracy
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
    ax2.set_xlabel('# epoch')
    ax2.set_ylabel('accuracy')
    ax2.tick_params('y')
    ax2.legend(loc='lower right', shadow=False)
    ax2.set_title('Model accuracy through #epochs', color=orange, fontweight='bold')
    plt.savefig("train_acc_loss_epochs.png")
    plt.show()




def build_model(input_dim, hidden_neurons, output_dim):
    """
    Construct, compile and return a Keras model which will be used to fit/predict
    """
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def untag(tagged_sentence):
    """ 
    Remove the tag for each tagged term.:param tagged_sentence: a POS tagged sentence
    :type tagged_sentence: list
    :return: a list of tags
    :rtype: list of strings
    """
    return [w for w, _ in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    """
    Split tagged sentences to X and y datasets and append some basic features.:param tagged_sentences: a list of POS tagged sentences
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return: 
    """
    X, y = [], [] 
    for pos_tags in tagged_sentences:
        for index, (term, class_) in enumerate(pos_tags):
            # Add basic NLP features for each sentence term
            X.append(add_basic_features(untag(pos_tags), index))
            y.append(class_)
    return X, y



def add_basic_features(sentence_terms, index):
    """ Compute some very basic word features.        :param sentence_terms: [w1, w2, ...] 
        :type sentence_terms: list
        :param index: the index of the word 
        :type index: int
        :return: dict containing features
        :rtype: dict
    """
    term = sentence_terms[index]
    return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }

CUSTOM_SEED=42
np.random.seed(CUSTOM_SEED)

re_pattern = re.compile("\(([a-z]+)\s(["u"\U00001000-\U0000109F""]+)+\)")


dir_path = "/home/harshil/Harshil/gt/spring2020/research2/ml-evaluation-models/alt-treebank"

data_path = "data/pos_data.json"
# original_data_path = "my-alt-190530/data"


with open(os.path.join(dir_path, data_path), "r") as data_f:
    data = json.load(data_f)

print("Dataset Information")
print("-"*50)
print("Num sentences: {}".format(len(data)))
tags = set([tag for sentence in data for _, tag in sentence])
print("Num tags: {}\nTags:{}".format(len(tags), tags))
all_words = [word for sentence in data for word, _ in sentence]
unique_words = set(all_words)
print("Num unique words: {}".format(len(unique_words)))
print("Total words: {}".format(len(all_words)))

train_test_cutoff = int(.80 * len(data)) 
training_sentences = data[:train_test_cutoff]
testing_sentences = data[train_test_cutoff:]
train_val_cutoff = int(.25 * len(training_sentences))
validation_sentences = training_sentences[:train_val_cutoff]
training_sentences = training_sentences[train_val_cutoff:]


#After transforming to dataset, a huge matrix is created with each row representing a data point in X train and the columns representing the features. 
X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(testing_sentences)
X_val, y_val = transform_to_dataset(validation_sentences)

dict_vectorizer = DictVectorizer(sparse=True)
dict_vectorizer.fit(X_train)# Learn feature names
X_train = dict_vectorizer.transform(X_train)
X_test = dict_vectorizer.transform(X_test)
X_val = dict_vectorizer.transform(X_val)

# Fit LabelEncoder with our list of classes
# Label encoder converts each class to a number. One hot encoding basically adds columns and sets the relevant column to 1.

#there was a feature difference so had to add this code
all_features = set(y_train).union(set(y_test))
feature_difference_train = all_features - set(y_train)
feature_difference_test = all_features - set(y_test)

# print(feature_difference_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_val)# Encode class values as integers
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)
y_train = np_utils.to_categorical(y_train, num_classes=12)
y_test = np_utils.to_categorical(y_test, num_classes=12)
y_val = np_utils.to_categorical(y_val, num_classes=12)


target_names = label_encoder.classes_
print(target_names)
print(len(target_names))


# print(y_val.shape)

model_params = {
    'build_fn': build_model,
    'input_dim': X_train.shape[1],
    'hidden_neurons': 512,
    'output_dim': y_train.shape[1],
    'epochs': 5,
    'batch_size': 256, #previously 256
    'verbose': 1,
    'validation_data': (X_val, y_val),
    'shuffle': True
}
custom_callbacks = [
    History(),
    CSVLogger('log.csv', append=True, separator=';')
 ]
clf = KerasClassifier(**model_params)
history = clf.fit(X_train, y_train, callbacks=custom_callbacks)

print(history.history)
plot_model_performance(
    train_loss=history.history["loss"],
    train_acc=history.history["accuracy"],
    train_val_loss=history.history["val_loss"],
    train_val_acc=history.history["val_accuracy"]
)
y_preds = clf.predict(X_test)
# print("Y PRED")
# print(y_preds[1])
# print(y_preds[2])

# print("Y test")
# print(y_test[1])
# print(y_test[2])
# print("Y test")
# print(y_test[1])
# print(y_test[2])
score = clf.score(X_test, y_test)
print(score)
y_test=np.argmax(y_test, axis=1)
classif_report = classification_report(y_true=y_test, y_pred=y_preds, target_names=target_names, labels=np.unique(y_preds))
print(np.shape(y_test))
print(np.shape(y_preds))
print(classif_report)
clf.model.save('keras_mlp.h5')
plot_model(clf.model, to_file='model.png', show_shapes=True)

with open("output.txt", "w") as out_f:
    out_f.write(classif_report)
    out_f.write("\n")
    out_f.write(str(history.history))




'''
Previous code

# with open(os.path.join(dir_path, original_data_path), "r") as data_f:
   # data = data_f.readlines()

# # print(len(data))
# # print(data[0])
# result = []
# for sent in data:
   # trimmed_sent = re_pattern.findall(sent)
   # result.append(trimmed_sent)



'''
