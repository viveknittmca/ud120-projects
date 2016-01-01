#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

clf = SVC(kernel="rbf", C=10000.0)
time_fit = time()
clf.fit(features_train, labels_train)
print "Training Time: " , (time()-time_fit) , "s"
time_pred = time()
pred = clf.predict(features_test)
print "Prediction Time:", (time() - time_pred), "s"
count = len([ "1" for id,label in enumerate(pred) if label == 1 ])
print "Count: " , count
#print "10: ", pred[10]
#print "26: ", pred[26]
#print "50: ", pred[50]
#print "Accuracy: %f" % accuracy_score(pred, labels_test)

#########################################################


