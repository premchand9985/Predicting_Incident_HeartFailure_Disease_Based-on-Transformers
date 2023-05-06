# -*- coding: utf-8 -*-
"""
Created on Sat Apr  15 13:45:39 2023

@author: premchand
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Display the first five rows of the dataset
heart_data.head()

# Display the last five rows of the dataset
heart_data.tail()

# Print the shape of the dataset (rows, columns)
heart_data.shape

# Display information about the dataset, such as data types and non-null counts
heart_data.info()

# Print the sum of null values in each column
heart_data.isnull().sum()

# Print the summary statistics for each column in the dataset
heart_data.describe()

# Print the target value count
print("The Target Value")
print("", heart_data['target'].value_counts())

# Define the feature matrix and target vector
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Print the feature matrix
print(X)

# Print the target vector
print(Y)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Print the shapes of the full dataset, training set, and testing set
print(X.shape, X_train.shape, X_test.shape)

# Instantiate the K-Nearest Neighbors model
knn_model = KNeighborsClassifier()

# Train the K-Nearest Neighbors model on the training set
knn_model.fit(X_train, Y_train)

# Make predictions on the training set
X_train_preds = knn_model.predict(X_train)
training_accuracy = accuracy_score(X_train_preds, Y_train)

# Print the training set accuracy
print('Accuracy on Training data : ', training_accuracy)

# Make predictions on the test set
X_test_preds = knn_model.predict(X_test)
test_accuracy = accuracy_score(X_test_preds, Y_test)

# Print the test set accuracy
print('Accuracy on Test data : ', test_accuracy)

# Define an example input for making predictions
input_data = (52, 1, 2, 258, 199, 1, 1, 162, 0, 0.5, 2, 0, 7)

# Convert the example input to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the input data for use with the K-Nearest Neighbors model
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction with the K-Nearest Neighbors model on the example input
prediction = knn_model.predict(input_data_reshaped)

# Print the prediction
print(prediction)

# Print the interpretation of the prediction
if (prediction[0] == 0):
    print('The Person has Heart failure')
elif (prediction[0] == 1):
    print('The Person has Myocardial infraction')
elif (prediction[0] == 2):
    print('The Person has Dilated cardiomyopathy')
elif (prediction[0] == 3):
    print('The Person has Coronary vasospasm')
elif (prediction[0] == 4):
    print('The Person has Atrial fibrillation')
else:
    print('The Person has Arrhythmia- abnormal heart rhythm')
