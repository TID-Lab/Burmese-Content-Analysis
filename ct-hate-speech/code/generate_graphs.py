import matplotlib.pyplot as plt
import json
from collections import defaultdict

def draw_graph_feature_extractor_classifier(data, graphs):
    x_labels = [100, 500, 1000, 5000, 10000]
    colors = ["red", "olive", "blue", "orange", "purple"]
    dict_g = defaultdict(lambda: defaultdict(list))
    for graph_key in graphs:
        for clf_key, clf in data.items():
            plt.figure()
            for i, (fex_key, fex) in enumerate(clf.items()):
                for data_set_size in sorted(fex["original"], key = lambda x: int(x["training_data_size"])):
                    dict_g[graph_key][fex_key].append(float(data_set_size[graph_key]))
                plt.plot(x_labels, fex_key, data=dict_g[graph_key], color=colors[i], linewidth=2, label = fex_key, marker = "o", markersize = 5)
            plt.legend()
            plt.title(graph_key.title() + " vs " + "Samples per class")
            plt.savefig("../graphs/" + clf_key +"_"+graph_key+".png")
            plt.show()



def main():
    with open("logs/svm_logs_compiled.json", "r") as logs:
        logs = json.load(logs)

        draw_graph_feature_extractor_classifier(logs, ["accuracy_avg", "recall_avg", "precision_avg"])
        
        
        


if __name__ == "__main__":
    main()


