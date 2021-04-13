# KNN_Scikit
This is implementation of KNN on IRIS Dataset using Scikit

Objective
In this project, we build a simple kNN model for a dataset and we explore some methods to optimize model parameters.
2- Dependencies
Python > 2.7
Scikit Learn package
Pandas toolkit (If it is needed)
Numpy toolkit (If it is needed)

3- Dataset
For this assignment, we use Iris dataset. More information about this data set can be found here: https://archive.ics.uci.edu/ml/datasets/iris

4- kNN project description

Part 1 – Preliminary Tasks (Simple data wrangling)
A- Make yourself familiar with data. You can read about it from the link above.
B- Make yourself familiar with “train_test_split” function of Scikit. You can find some basic information here:
https://scikitlearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Part 2 – Building and training the kNN model
A- Split your data into training and test dataset using “train_test_split” function. Put test_size = 0.2 and use stratify=y.
B- Use k value from the input (in the next section, we try to find optimum k) and build your model. For example, if you pick your k = 4, 3 out of 4 is the winner!
C- Two main functions that should be used here are: “KNeighborsClassifier” for knn classifier and “fit” method of the “KNeighborsClassifier” object for fitting your test and training datasets. This is your training step!

Part 3 – Testing kNN Model
A- To test your model use “predict” method of “KNeighborsClassifier” and then use the “score” method of “KNeighborsClassifier” to get the accuracy of your model.
Note: This method of testing is called holdout.
B- Report the accuracy.

Part 4 – Cross validation
A- In Cross-Validation dataset is randomly split up to T groups. Then one of the groups is considered as the test and the rest are used as the training. The process is repeated for all T groups.
For more information on cross validation in scikit please see:
https://scikit-learn.org/stable/modules/cross_validation.html
B- To run cross validation, use “cross_val_score” function. The “cross value” parameter will be read from input.

Part 5 - Optimizing n-neighbor parameter
A- Optimizing a model parameters (hyper-tuning parameters) is a process to find optimal parameters to improve model accuracy.
B- For optimizing k value you need to use “GridSearchCV” function with given range of k. in this case test your model with k in the range of [1 .. 25]. For more information on “GridSearchCV” please see:
https://scikitlearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
C- There are at least two types of weights (distance) can be used in scikit. Investigate both of them and find which one performs better.
