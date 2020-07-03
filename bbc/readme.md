## Instructions

- Make sure that data folder contains the training and the test data. The data can be obtained from the lab's google drive under ml-evaluation-models/evaluation.
- To train and test the classifier, run the following command:
```python
python3 supervised-2-class.py -m <nb|svm> -x <X data file> -y <y data file> -c <corpus file> -v <tfidf|ngrams> -n <1|2>
```
- The corpus file can be the same as the X data file or the actual corpus file from lab's google drive. 
- The output will be generated in the logs folder.

- If instead, you would like to run the files on PACE cluster, `ssh` into the login node, and follow these commands from the `code` folder of the project after editing the pace script as you like. The logs by default will be generated in the logs folder and the terminal output will be available at the root of your login node.


```
qsub pace_script.pbs
```
- The classifier will be saved in the project directory.


