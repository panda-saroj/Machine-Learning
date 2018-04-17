# Machine-Learning
Machine Learning Repository.

This repository contains implementation of various Machine Learning Algorithms.
Please free to contact me on sarojpanda.blr@gmail.com, if you need any information on these.



Naive_Bayes_ML.py
1. Implements Naive Bayes Algorithm. 
2. Uses the mushroom data-set from UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/datasets.html) stored in the file agaricus-lepiota as comma separated values. 
3. Use the top 4000 instances for training and the rest for testing.
4. Computes the Accuracy Of The Model and Contigency Table



KNN_Classify.py
1. Implements a KNN Classifier Function myknnclassify(X, test, k), which predicts the class of the test instance test based on training data in X and the k value 
 i. X: File containing the data set
 ii. test: Test Instance
 iii. k: k value
2. Uses a two dimensional points dataset points.csv for testing. You are free to use your dataset and k value
3. Modify the following part to run for your test values.

#Testing with points data

X = 'points.csv'

test = [33,3]

k = 6


myknnclassify(X, test, k)



KNN_Regression.py
1. Implements a KNN Regression Function myknnregress(X, test, k), which predicts the regression value for the test instance test based on training data in X and the k value. 
 i. X: File containing the data set
 ii. test: Test Instance
 iii. k: k value
2. Uses a two dimensional points dataset points.csv for testing. You are free to use your dataset and k value
3. Modify the following part to run for your test values.

#Testing with points data

X = 'points.csv'

test = [33,3]

k = 10

myknnregress(X, test, k)



For The following programs, we need 
Softwares
====================
1. Python 3.5 or more
2. python numpy module
3. python sklearn module

Binary_Logistic Regression.py
1. Implements a simple perceptron Neural Network for logistric regression.
2.  #Mean and covariance of the dataset of two classes
mean1 = [1, 0]
mean2 = [0, 1.5]

cov1 = [[1, 0.75], [0.75, 1]]
cov2 = [[1, 0.75], [0.75, 1]]

3. Training data, generate 3000 training instances in two sets of random data points (1500 in each) and label them 0 and 1.
4. Generate testing data in the same manner but sample 500 instances for each class,
i.e., 1000 in total.
5. Use sigmoid function for your activation function and cross entropy for your objective
function, and perform batch training.
6. Plot an ROC curve and compute Area Under the Curve (AUC) in
the end to evaluate your implementation
7. Change the following parameters at the top of the file for different number of iterations and learning rate

train_iterations = 3000

learning_rate = 0.001
