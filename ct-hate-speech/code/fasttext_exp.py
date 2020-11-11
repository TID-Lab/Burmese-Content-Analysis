# import json
# import csv
# with open("../data/X_filtered.json", "r") as data_in:
#     data = json.load(data_in)
# len(data)
# with open("../data/y_filtered.json", "r") as data_in:
#     data_y = json.load(data_in)
# len(data_y)
from sklearn.model_selection import train_test_split

# from collections import Counter

# print("Length of original data " + str(len(data)))
# with open("../data/fasttext/data.all.txt", "w") as out_all:
#     for sample, label in zip(data, data_y):
#         if len(sample.strip()) != 0:
#             prt_str = "__label__{} {}\n".format(label, sample)
#             out_all.write(prt_str)



# def upsampling(input_file, output_file, ratio_upsampling=1):
#     #
#     # Create a file with equal number of tweets for each label
#     #    input_file: path to file
#     #    output_file: path to the output file
#     #    ratio_upsampling: ratio of each minority classes vs majority one. 1 mean there will be as much of each class than there is for the majority class.
 
#     i=0
#     counts = {}
#     dict_data_by_label = {}# GET LABEL LIST AND GET DATA PER LABEL
#     with open(input_file, 'r') as infile:
#         data = infile.readlines();
#         print("length of data " + str(len(data)))
#         for row in data:
#             counts[row.split()[0]] = counts.get(row.split()[0], 0) + 1
#             if not row.split()[0] in dict_data_by_label:
#                 dict_data_by_label[row.split()[0]]=[row]
#             else:
#                 dict_data_by_label[row.split()[0]].append(row)
#             i=i+1
#             if i%2000 ==0:
#                 print("read" + str(i))
#                 print(counts)
#     # FIND MAJORITY CLASS
#     majority_class=""
#     count_majority_class=0
#     for item in dict_data_by_label:
#         if len(dict_data_by_label[item])>count_majority_class:
#             majority_class= item
#             count_majority_class=len(dict_data_by_label[item])  
    
#     # UPSAMPLE MINORITY CLASS
#     data_upsampled=[]
#     for item in dict_data_by_label:
#         data_upsampled.extend(dict_data_by_label[item])
#         if item != majority_class:
#             items_added=0
#             items_to_add = count_majority_class - len(dict_data_by_label[item])
#             while items_added<items_to_add:
#                 data_upsampled.extend(dict_data_by_label[item][:max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))])
#                 items_added = items_added + max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))
#     print("Length of Upsampled data " + str(len(data_upsampled)))
#     with open('../data/fasttext/data.upsampled.all.txt', 'w') as f:
#         f.writelines(data_upsampled)
# upsampling("../data/fasttext/data.all.txt", "../data/fasttext/data.upsampled.all.txt")

with open("../data/fasttext/data.upsampled.all.txt", "r") as infile:
    data_in = infile.readlines()
    dataY = [x.split()[0] for x in data_in]
    dataX = [" ".join(x.split()[1:]) for x in data_in]

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY,
                                                stratify=dataY, 
                                                test_size=0.20)

with open("../data/fasttext/data.train.oversampled.txt", "w") as out_train:
    for sample, label in zip(X_train, y_train):
        prt_str = "{} {}\n".format(label, sample)
        out_train.write(prt_str)
with open("../data/fasttext/data.test.oversampled.txt", "w") as out_train:
    for sample, label in zip(X_test, y_test):
        prt_str = "{} {}\n".format(label, sample)
        out_train.write(prt_str)


import fasttext
model = fasttext.train_supervised(input="../data/fasttext/data.train.oversampled.txt")
model.save_model("../classifiers/fasttext-hate-clf-oversampled.bin")
model.test("../data/fasttext/data.test.oversampled.txt")
#import fasttext.util
#model = fasttext.train_supervised(input="../data/fasttext/data.train.txt", pretrainedVectors = "../data/fasttext/cc.my.300.vec", dim = 300)
from sklearn import metrics
y_pred = model.predict(X_test)
y_pred_pro = [x[0] for x in y_pred[0]]         
print(metrics.classification_report(y_test, y_pred_pro))

