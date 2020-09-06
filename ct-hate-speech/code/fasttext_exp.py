import json
with open("X_filtered.json", "r") as data_in:
    data = json.load(data_in)
len(data)
with open("y_filtered.json", "r") as data_in:
    data_y = json.load(data_in)
len(data_y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, data_y,
                                                stratify=y, 
                                                test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(data, data_y,
                                                stratify=data_y, 
                                                test_size=0.20)
len(X_train)
len(X_test)
from collection import Counter
from collections import Counter
Counter(y_train)
Counter(y_test)
44/1086
178/4338
with open("data.train.txt", "w") as out_train:
    for sample, label in zip(X_train, y_train):
        prt_str = "__label__{} {}\n".format(label, sample)
        out_train.write(prt_str)
ls
head data.train.txt
pwd
with open("data.test.txt", "w") as out_train:
    for sample, label in zip(X_test, y_test):
        prt_str = "__label__{} {}\n".format(label, sample)
        out_train.write(prt_str)
import fasttext
model = fasttext.train_supervised(input="data.train.txt")
model.save_model("../classifiers/fasttext-hate-clf.bin")
model.test("data.test.txt")
fasttext.util.download_model('my', if_exists='ignore')
import fasttext.util
fasttext.util.download_model('my', if_exists='ignore')
model = fasttext.train_supervised(input="data.train.txt", pretrainedVectors = "cc.my.300.bin")
model = fasttext.train_supervised(input="data.train.txt", pretrainedVectors = "cc.my.300.bin", dim = 300)
model = fasttext.train_supervised(input="data.train.txt", pretrainedVectors = "cc.my.300.vec", dim = 300)
%save fasttext-exp
%save fasttext-exp.py
lsmagic
save fasttext-exp
%history -f fasttext-exp.py
