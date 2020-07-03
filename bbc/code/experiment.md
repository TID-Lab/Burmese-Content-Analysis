## Classifiers
### BBC Dataset
#### Support Vector Machines:

##### Tunable parameters: 
###### C (Regularization) 
* Controls how big the margin is between the support vector points and the hyperplane. We get smaller margins for larger value of C ensuring that the training data is well classified. Smaller margins may be bad for unseen test data. 
* Smaller values for C give us large margins which can lead to some misclassified data points but generalizes well for unseen data.

* I tested values C = [0.1, 1, 10, 100] with C = 1 resulting in 91% accuracy. Sometimes, C = 100 performs better. So need to run it multiple times to find the average.


###### Gamma 
* Controls the radius of influence of the support vectors. Larger value means that the radius of influence is support vector itself leading to overfitting. Smaller values means that SVM cannot capture the shape of the data. 

* I tested values Gamma = [1, 0.1, 0.01] with Gamma = 1 resulting in highest accuracy.

###### Kernel
* This has to be the Gaussian Kernel. This can be shown by plotting the 3 major dimesnions selected using PCA on a graph to see that the data cannot be linearly separated.


###### Training size and Testing size
* 50-50?, 70-30?, 80-20? Currently using 80-20 train-test split.


#### Multinomial Naive Bayes Classifier:

**Why Multinomial?**
* When data that is ingested in the model contains frequency information, Multinomial Naive Bayes classifer usually performs better. Here, we input the TfIdf vector matrix of all the documents in out training set into the classifier. 

##### Tunable parameters:

###### Alpha
* Laplace smoothing parameter to ensure that if a word and a class do not appear together in the training dataset, the probability of the document belonging to that particular class is not zero. 
* I tested values Alpha = [0.01, 0.05, 0.1, 0.3, 0.5] Alpha = 0.1 resulting in best accuracy.

###### Training size and Testing size 
* 50-50? 70-30? 80-20? Currently using 80-20 train-test split.

	
	
###### How were the optimal parameters chosen?

* In ML, the generalization curve generally follow a U shaped curve where simple hyper parameters that are not able to capture all patterns in the training data lead to high generalization error (underfitting) and complex hyper parameters lead to overfitting the training data causing high generalization error. Hence, the generalization error is high on both ends of the U shaped curve because of either underfitting or overfitting. The goal was to find a point in between that results in the best accuracy. For both SVM and NB, I initially formed a range for each hyperparameter. If the optimal hyperparameter was in the middle of that range with the end values giving high generalization error, I can conclude the the middle value of the range yields the minima on the generalization curve. If the optimal value ended up being on one of the ends of the range, I would readjust the range until I find the optimal hyperparameters that lie in the middle of the range. This ensures that the hyperparameters I selected are indeed optimal.


