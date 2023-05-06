# -*- coding: utf-8 -*-
"""
Created on Thu Apr  13 09:12:05 2023

@author: premchand
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics


# Load the dataset
heart_data = pd.read_csv('framingham.csv')

# Display the first 5 rows
heart_data.head()

# Display the last 5 rows
heart_data.tail()

# Display the shape of the dataset
heart_data.shape

# Display dataset information
heart_data.info()

# Display the number of missing values for each column
heart_data.isnull().sum()

# Display summary statistics
heart_data.describe()

# Separate features (X) and target (Y) variables
X = heart_data.drop('TenYearCHD', axis=1)
Y = heart_data['TenYearCHD']

# Fill missing values with column means
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)

# Check if there are any missing values left
X.isnull().sum(axis=0)

print(X)
print(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Print the shapes of the original, training, and testing datasets
print(X.shape, X_train.shape, X_test.shape)

# Instantiate the Logistic Regression model
logreg_model = LogisticRegression()

# Train the model on the training set
logreg_model.fit(X_train, Y_train)

# Make predictions on the training set
X_train_preds = logreg_model.predict(X_train)
training_accuracy = accuracy_score(X_train_preds, Y_train)

# Print the training set accuracy
print('Accuracy: ', training_accuracy)

# Make predictions on the test set
X_test_preds = logreg_model.predict(X_test)
test_accuracy = accuracy_score(X_test_preds, Y_test)

# Print the test set accuracy
print('Accuracy on test data : ', test_accuracy)

# Import additional metrics for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Print the classification report for Logistic Regression
print('Classification Report is: \n \n', classification_report(Y_test, X_test_preds))

# Print the accuracy score for Logistic Regression
print('The accuracy score is:', accuracy_score(Y_test, X_test_preds))

# Print the confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(Y_test, X_test_preds)
print('\n Confusion matrix \n \n', cm_logreg)

# Print the classification report for Logistic Regression
print(classification_report(Y_test, X_test_preds))

# Plot the confusion matrix for Logistic Regression using the old method
#plot_confusion_matrix(logreg_model, X_test, Y_test)
plt.show()

# Plot the confusion matrix for Logistic Regression using the new method
disp_logreg = ConfusionMatrixDisplay(cm_logreg, display_labels=logreg_model.classes_)
disp_logreg.plot()
plt.show()

# Calculate and print the area under the ROC curve for Logistic Regression
roc_auc_logreg = metrics.roc_auc_score(Y_test, X_test_preds)
print(roc_auc_logreg)

# Plot the ROC curve for Logistic Regression
plt.figure(figsize=(10,8))
probas_logreg = logreg_model.predict_proba(X_test)
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(Y_test, probas_logreg[:, 1])
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

plt.plot(fpr_logreg, tpr_logreg, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc_logreg))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Instantiate the Decision Tree Classifier model
dt_model = DecisionTreeClassifier()

# Train the model on the training set
dt_model.fit(X_train, Y_train)

# Make predictions on the training set
X_train_preds_dt = dt_model.predict(X_train)
training_accuracy_dt = accuracy_score(X_train_preds_dt, Y_train)

# Print the training set accuracy for Decision Tree Classifier
print('Accuracy DecisionTree: ', training_accuracy_dt)

# Make predictions on the test set
X_test_preds_dt = dt_model.predict(X_test)
test_accuracy_dt = accuracy_score(X_test_preds_dt, Y_test)

# Print the test set accuracy for Decision Tree Classifier
print('Accuracy on DecisionTree test data : ', test_accuracy_dt)

# Print the classification report for Decision Tree Classifier
print('Classification DecisionTree Report is: \n \n', classification_report(Y_test, X_test_preds_dt))

# Print the accuracy score for Decision Tree Classifier
print('The accuracy score is:', accuracy_score(Y_test, X_test_preds_dt))

# Print the confusion matrix for Decision Tree Classifier
cm_dt = confusion_matrix(Y_test, X_test_preds_dt)
print('\n Confusion matrix \n \n', cm_dt)

# Print the classification report for Decision Tree Classifier
print(classification_report(Y_test, X_test_preds_dt))

# Plot the confusion matrix for Decision Tree Classifier using the old method
#plot_confusion_matrix(dt_model, X_test, Y_test)
plt.show()

# Plot the confusion matrix for Decision Tree Classifier using the new method
disp_dt = ConfusionMatrixDisplay(cm_dt, display_labels=dt_model.classes_)
disp_dt.plot()
plt.show()

# Calculate and print the area under the ROC curve for Decision Tree Classifier
roc_auc_dt = metrics.roc_auc_score(Y_test, X_test_preds_dt)
print(roc_auc_dt)

# Plot the ROC curve for Decision Tree Classifier
plt.figure(figsize=(10, 8))
probas_dt = dt_model.predict_proba(X_test)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(Y_test, probas_dt[:, 1])
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.plot(fpr_dt, tpr_dt, lw=1, label='ROC fold DecisionTree (area = %0.2f)' % (roc_auc_dt))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate DecisionTree')
plt.ylabel('True Positive Rate DecisionTree')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
