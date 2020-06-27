#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:09:57 2020

@author: Xinyu Zhang
"""



#Question 1
#import all libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#load data 
loan = pd.read_csv('loan_data_part_1.csv')
loan.loc[loan["loan_status"]=="Fail", "loan_status"] = 1
loan.loc[loan["loan_status"]=="Current", "loan_status"] = 0
x = loan.iloc[:, 1:36]
y = loan.iloc[:, 0]

#standardize data
scale = StandardScaler()
x_scale = scale.fit_transform(x)


#traning and making predictions
#evaluate the performance of LDA, randomly try 10 times
'''
print(cm)
print('Accuracy' + str(accuracy_score(y, y_pred)))
true_positive = cm[0,0]/(cm[0,0] + cm[0,1])
true_negative = cm[1,1]/(cm[1,1] + cm[0,1])
print("The true positive is {} and the true negative is {}.".format(true_positive, true_negative))
'''

# question1
#KNN model
knn = KNeighborsClassifier(n_neighbors=50)
#train model with cv of 10 
y_pred = cross_val_predict(knn, x_scale, y, cv=10)
#create confusion matrix
cm_knn = confusion_matrix(y, y_pred)
#random forest 
rf = RandomForestClassifier(n_estimators = 100, random_state = 0)
y_pred__rf = cross_val_predict(rf, x_scale, y, cv=10)
cm_rf = confusion_matrix(y, y_pred__rf)
#LDA
lda = LinearDiscriminantAnalysis()
y_pred__lda = cross_val_predict(lda, x_scale, y, cv=10)
cm_lda = confusion_matrix(y, y_pred__lda)

#Question 2
#undersample
rus = RandomUnderSampler(random_state=0)
x_resampled, y_resampled = rus.fit_resample(x_scale, y)
x_resampled = scale.fit_transform(x_resampled)

knn = KNeighborsClassifier(n_neighbors=50)
#train model with cv of 10 
y_pred_knn_us = cross_val_predict(knn, x_resampled, y_resampled, cv=10)
#create confusion matrix
cm_knn_us = confusion_matrix(y_resampled, y_pred_knn_us)
#random forest 
rf = RandomForestClassifier(n_estimators = 100, random_state = 0)
y_pred__rf_us = cross_val_predict(rf, x_resampled, y_resampled, cv=10)
cm_rf_us = confusion_matrix(y_resampled, y_pred__rf_us)
#LDA
lda = LinearDiscriminantAnalysis()
y_pred__lda_us = cross_val_predict(lda, x_resampled, y_resampled, cv=10)
cm_lda_us = confusion_matrix(y_resampled, y_pred__lda_us)

#Question 6
# import library for logistic
logis = LogisticRegression(penalty='l1', C=10, random_state=0, solver='liblinear')
y_pred_logis_LASSO = cross_val_predict(logis, x_resampled, y_resampled, cv=10)
cm_logis_LASSO = confusion_matrix(y_resampled, y_pred_logis_LASSO)


#Question 7
#part 1
logis = LogisticRegression(penalty='l1', C=10, random_state=0, solver='liblinear')
logis.fit(x_resampled, y_resampled)
y_pred_LASSO_noCV = logis.predict(x_resampled)
cm_LASSO_noCV = confusion_matrix(y_resampled, y_pred_LASSO_noCV)

#part2
coefficients = []
#set a set of alpha
lasso_alphas = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
#for loop to try different value of alpha
for a in lasso_alphas:
    logis = LogisticRegression(penalty='l1', C=a, random_state=0, solver='liblinear')
    logis.fit(x_resampled, y_resampled)
    coefficients.append(logis.coef_)
#plot result
'''
plt.figure(figsize=(20, 6))
plt.subplot(121)
ax = plt.gca()
ax.plot(lasso_alphas, coefficients)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
'''
#sift alpha with 10 parameters
#round data
"""
for coef in coefficients:
    for i in range(34):
        coef[i] = round(coef[i], 5)
    print(coef)
"""   
#calculate number of non-zeros
summ = []
for coef in coefficients:
    subsum = 0
    for i in range(34):
        if coef[0,i] != 0:
            subsum += 1
    summ.append(subsum)
print(summ)

#run another for loop to narrow down the value of lambda
alphas2 = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]
coefficients2 = []
for a in alphas2:
    logis = LogisticRegression(penalty='l1', C=a, random_state=0, solver='liblinear')
    logis.fit(x_resampled, y_resampled)
    coefficients2.append(logis.coef_)
summ2 = []
for coef in coefficients2:
    subsum = 0
    for i in range(34):
        if coef[0,i] != 0:
            subsum += 1
    summ2.append(subsum)
print(summ2)

#transform an array into a list
l = []
for element in coefficients2[1]:
    l.append(element)
    
#find the position of non-zero element in coefficient[0]
position = []
for i in range(34):
    if l[0][i] != 0:
        position.append(i)
print(position)

#find column names in x_sampled
attributes = []
for item in position:
    attributes.append(x.columns.values[item])
print(attributes)

#Question 3b
#reduced data set
loan_reduced = loan[["loan_status","annual_inc", "emp_length","int_rate",
      "loan_amnt", "num_tl_90g_dpd_24m","percent_bc_gt_75",
      "pub_rec_bankruptcies","sub_grade","term",
      "verification_status"]]
x_reduced = loan.iloc[:, 1:11]
y_reduced = loan.iloc[:, 0]
#undersample
rus = RandomUnderSampler(random_state=0)
x_reduced_resampled, y_reduced_resampled = rus.fit_resample(x_reduced, y_reduced)
x_reduced_resampled = scale.fit_transform(x_reduced_resampled)

#logistic
logis = LogisticRegression(random_state=0)
#logis.fit(x_reduced_resampled, y_reduced_resampled)
y_pred_reduced_logis = cross_val_predict(logis, x_reduced_resampled, y_reduced_resampled, cv=10)
cm_reduced_logis = confusion_matrix(y_reduced_resampled, y_pred_reduced_logis)

#50-NN
knn = KNeighborsClassifier(n_neighbors=50)
y_pred_reduced_50NN = cross_val_predict(knn, x_reduced_resampled, y_reduced_resampled, cv=10)
cm_reduced_50NN = confusion_matrix(y_reduced_resampled, y_pred_reduced_50NN)

#Random Forest
rf = RandomForestClassifier(n_estimators = 100, random_state = 0)
y_pred_reduced_rf = cross_val_predict(rf, x_reduced_resampled, y_reduced_resampled, cv=10)
cm_reduced_rf = confusion_matrix(y_reduced_resampled, y_pred_reduced_rf)

#Question 4
#50-NN
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_reduced_resampled, y_reduced_resampled)
y_pred_knn_noCV = knn.predict(x_reduced_resampled)
knn_proba = knn.predict_proba(x_reduced_resampled)
fpr, tpr, thresholds = roc_curve(y_reduced_resampled, knn_proba[:,1])
roc_auc = metrics.auc(fpr, tpr)
#plot ROC for KNN
plt.title('Receiver Operating Characteristic KNN')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_reduced_resampled, y_reduced_resampled)
y_pred_lda_noCV = lda.predict(x_reduced_resampled)
lda_proba = lda.predict_proba(x_reduced_resampled)
fpr, tpr, thresholds = roc_curve(y_reduced_resampled, lda_proba[:,1])
roc_auc = metrics.auc(fpr, tpr)
#plot
plt.title('Receiver Operating Characteristic LDA')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#Part 2 Question11
#dataset a
#load new dataset
loan2_a = pd.read_csv('loan_data_part_2_a.csv')
loan2_a.loc[loan2_a["loan_status"]=="Fail", "loan_status"] = 1
loan2_a.loc[loan2_a["loan_status"]=="Current", "loan_status"] = 0
x2_a = loan.iloc[:, 1:36]
y2_a = loan.iloc[:, 0]

#logistic
logis = LogisticRegression(random_state=0)
logis.fit(x_resampled, y_resampled)
y_pred_logis_oos_a = logis.predict(x2_a)
cm_logis_oos_a = confusion_matrix(y2_a, y_pred_logis_oos_a)

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_resampled, y_resampled)
y_pred_lda_oos_a = lda.predict(x2_a)
cm_lda_oos_a = confusion_matrix(y2_a, y_pred_lda_oos_a)

#50-NN
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_resampled, y_resampled)
y_pred_knn_oos_a = knn.predict(x2_a)
cm_knn_oos_a = confusion_matrix(y2_a, y_pred_knn_oos_a)

#dataset b
loan2_b = pd.read_csv('loan_data_part_2_b.csv')
loan2_b.loc[loan2_b["loan_status"]=="Fail", "loan_status"] = 1
loan2_b.loc[loan2_b["loan_status"]=="Current", "loan_status"] = 0
x2_b = loan.iloc[:, 1:36]
y2_b = loan.iloc[:, 0]

#logistic
logis = LogisticRegression(random_state=0)
logis.fit(x_resampled, y_resampled)
y_pred_logis_oos_b = logis.predict(x2_b)
cm_logis_oos_b = confusion_matrix(y2_b, y_pred_logis_oos_b)

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_resampled, y_resampled)
y_pred_lda_oos_b = lda.predict(x2_b)
cm_lda_oos_b = confusion_matrix(y2_b, y_pred_lda_oos_b)

#50-NN
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_resampled, y_resampled)
y_pred_knn_oos_b = knn.predict(x2_b)
cm_knn_oos_b = confusion_matrix(y2_b, y_pred_knn_oos_b)








    

        


