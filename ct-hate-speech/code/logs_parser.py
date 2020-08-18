import json
from os import listdir
from os.path import isfile, join
from collections import defaultdict


log_dir = "/home/harshil/Harshil/gt/spring2020/research2/burmese-NLP/ct-hate-speech/code/logs"  
log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
def get_data(file_name, file_text, samples_per_class):
    lines = file_text.readlines()
    for line in lines:
        line = line.rstrip("\n")
        if "weighted avg" in line:
            metrics = line.split("     ")
        if "Best Parameters" in line:
            best_params = line.split("     ")[1]
        if "Training data size" in line:
            training_data_size = line.split("     ")[1].split(":")[1]
        if "Testing data size" in line:
            testing_data_size = line.split("     ")[1].split(":")[1]
    return {
            "file_name" : file_name, 
            "training_data_size":training_data_size, 
            "testing_data_size":testing_data_size,
            "accuracy_avg" :metrics[3], 
            "recall_avg":metrics[2], 
            "precision_avg":metrics[1],  
            "best_params":best_params
            } 


classifiers_dict = defaultdict(lambda : defaultdict(lambda: defaultdict(list)))
for log_file in log_files:
    print(log_file)
    log_details = log_file.split("_")
    if log_details[0] == "nb" or log_details[0] == "svm":
        if log_details[1] == "ngrams":
            if log_details[-1] == "dedup":
                with open(join(log_dir, log_file), "r") as file_text:
                    training_data_samples_per_class = log_details[5] if 5 < len(log_details) else "default"
                    data = get_data(log_file, file_text, training_data_samples_per_class)
                    classifiers_dict[log_details[0]][log_details[1]+log_details[2]]["dedup"].append(data)
            else:
                with open(join(log_dir, log_file), "r") as file_text:
                    training_data_samples_per_class = log_details[4] if 4 < len(log_details) else "default"
                    data = get_data(log_file, file_text, training_data_samples_per_class)
                    classifiers_dict[log_details[0]][log_details[1]+log_details[2]]["original"].append(data)

        elif log_details[1] == "tfidf":
            if log_details[-1] == "dedup":
                with open(join(log_dir, log_file), "r") as file_text:
                    training_data_samples_per_class = log_details[5] if 5 < len(log_details) else "default"
                    data = get_data(log_file, file_text, training_data_samples_per_class)
                    classifiers_dict[log_details[0]][log_details[1]]["dedup"].append(data)
            else:
                with open(join(log_dir, log_file), "r") as file_text:
                    training_data_samples_per_class = log_details[4] if 4 < len(log_details) else "default"
                    data = get_data(log_file, file_text, training_data_samples_per_class)
                    classifiers_dict[log_details[0]][log_details[1]]["original"].append(data)




print(json.dumps(classifiers_dict, indent = 4))
with open("{}_logs_compiled.json".format(log_details[0]), "w") as out:
    json.dump(classifiers_dict, out, indent=4)
