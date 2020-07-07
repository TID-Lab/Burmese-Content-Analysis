## Instructions

- Make sure that data folder contains the training and the test data. The data can be obtained from the lab's google drive under ml-evaluation-models/evaluation/pos_data.json.
- To train and test the classifier, run the following command:
```python
python3 nn.py
```
- The output will be generated in the code folder.

- If instead, you would like to run the files on PACE cluster, `ssh` into the login node, and follow these commands from the `code` folder of the project after editing the pace script as you like. The logs by default will be generated in the `code` folder and the terminal output will be available at the root of your login node.


```
qsub pace_script.pbs
```

- The classifier will be saved in the `code` folder.


