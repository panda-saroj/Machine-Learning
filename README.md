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
1. Implements a KNN Classifier Function myknnclassify(X, test, k), 
 i. X: File containing the data set
 ii. test: Test Instance
 iii. k: k value
2. Uses a two dimensional points dataset for testing. You are free to use your dataset and k value
3. Modify the following part to run for your test values.

#Testing with points data
X = 'points.csv'
test = [33,3]
k = 6

myknnclassify(X, test, k)
