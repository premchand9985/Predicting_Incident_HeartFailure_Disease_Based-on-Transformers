# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:18:36 2023

@author: premchand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# for plots to appear inside the notebook


# for data preprocessing
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from mlxtend.classifier import StackingCVClassifier

# Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

# Read the heart dataset
heart_df = pd.read_csv("heart.csv")

# Display the first five rows of the dataset
print("", heart_df.head())

# Display the last five rows of the dataset
print("", heart_df.tail())

# Print the shape of the dataset
print("", heart_df.shape)

# Print the information about the dataset
print("", heart_df.info())

# Print the number of missing values in each column
print("", heart_df.isna().sum())

# Print the descriptive statistics of the dataset
print("", heart_df.describe().T)

# Finding the values
heart_df.target.value_counts()

# Plotting a bar chart of the target column
target_fig = heart_df.target.value_counts().plot(kind="bar", color=["salmon", "lightgreen"])
target_fig.set_xticklabels(labels=['Has heart disease', "Doesn't have heart disease"], rotation=0)
plt.title("Heart Disease values")
plt.ylabel('Amount')

# Visualizing the target column in a pie chart
labels = "Has heart disease", "Doesn't have heart disease"
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(heart_df.target.value_counts(), explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.show()

# Count the number of males and females in the dataset
heart_df.sex.value_counts()

# Plotting a bar chart of the sex column
sex_fig = heart_df.sex.value_counts().plot(kind="bar", color=["lightskyblue", "orange"])
sex_fig.set_xticklabels(labels=['Male', "Female"], rotation=0)
plt.title("Sex vs count")

# Visualizing the sex column in a pie chart
labels = 'Male', 'Female'
explode = (0, 0.06)

fig1, ax1 = plt.subplots()
ax1.pie(heart_df.sex.value_counts(), explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

# Compare the target column with the sex column
pd.crosstab(heart_df.target, heart_df.sex)

# Create a bar plot of the crosstab between target and sex columns
sex_target_fig = pd.crosstab(heart_df.target, heart_df.sex).plot(kind='bar', figsize=(10, 6), color=['orange', 'lightskyblue'])
sex_target_fig.set_xticklabels(labels=["Doesn't have heart disease", "Has heart disease"], rotation=0)

plt.legend(['Female', 'Male'])
plt.title("Heart Disease Frequency for Sex")
plt.ylabel("Count")

# Count the number of people with each type of chest pain
heart_df.cp.value_counts()

# Plotting a bar chart of the chest pain (cp) column
cp_fig = heart_df.cp.value_counts().plot(kind="bar", color=["salmon", 'orange', 'lightblue', "lightgreen"])
cp_fig.set_xticklabels(labels=['Typical angina', 'Atypical angina', 'Non-anginal', 'Asymptomatic'], rotation=0)
plt.title("Chest Pain Type")

# Preprocessing the data
X = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the models
logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
svm_model = svm.SVC()
stack = StackingCVClassifier(classifiers=[logreg, knn, dt, rf, ab, svm_model], meta_classifier=logreg)

# Training and evaluating the models
models = {
    "Logistic Regression": logreg,
    "K-Nearest Neighbors": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "AdaBoost": ab,
    "SVM": svm_model,
    "Stacking": stack
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test)}")
# Create a function for model evaluation
def evaluate_model(model, X_test, y_test):
    y_preds = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Precision Score:", precision_score(y_test, y_preds))
    print("Recall Score:", recall_score(y_test, y_preds))
    print("F1 Score:", f1_score(y_test, y_preds))

# Evaluate each model
for name, model in models.items():
    print(f"{name}:")
    evaluate_model(model, X_test, y_test)
    print("------------------------------------------------")

# Plot ROC curve for each model
plt.figure(figsize=(10, 6))
for name, model in models.items():
    plot_roc_curve(model, X_test, y_test, name=name)

plt.title("ROC Curves for Different Models")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend(loc="lower right")
plt.show()
