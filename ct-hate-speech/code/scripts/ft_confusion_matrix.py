import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def parse_labels(path):
    with open(path, 'r') as f:
        return np.array(list(map(lambda x: x[9:], f.read().split())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display confusion matrix.')
    parser.add_argument('test', help='Path to test labels')
    parser.add_argument('predict', help='Path to predictions')
    args = parser.parse_args()
    test_labels = parse_labels(args.test)
    pred_labels = parse_labels(args.predict)
    print(classification_report(test_labels, pred_labels))
    eq = test_labels == pred_labels
    print("Accuracy: " + str(eq.sum() / len(test_labels)))
    print(confusion_matrix(test_labels, pred_labels))
