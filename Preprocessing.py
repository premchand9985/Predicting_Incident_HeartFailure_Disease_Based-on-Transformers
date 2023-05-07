# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:12:16 2023

@author: premchand
"""

import re

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


import numpy as np


import pandas as pd

import os, sys, argparse

from matplotlib import pyplot as plt

models = [LogisticRegression(solver='lbfgs', max_iter=1000),  GaussianNB(),  DecisionTreeClassifier(), KNeighborsClassifier()]

DATA_DIR = "heart.csv"

def get_data(data_dir):
    df = pd.read_csv(data_dir)

    male = df.loc[df.sex == 1]
    female = df.loc[df.sex == 0]

    return df, male, female


def disease_percents(patients):
    wit = patients[patients.target == 1]
    without = patients[patients.target == 0]

    wit = (len(wit)/len(patients)) * 100
    without = (len(without)/len(patients)) * 100

    return wit, without

def numb_sex(males, females, total):
    numbMales = (len(males)/len(total))*100
    numbFemales = (len(females)/len(total))*100

    return numbMales, numbFemales

def create_sets(data):
    x = data.drop('target', axis=1)
    y = data.target

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_split = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_split, y, test_size=0.3)

    array = data.values
    X = array[:,0:13]
    print(X)
    Y = array[:,13]

    return X_train, X_test, y_train, y_test, X, Y


def train(x_train, x_test, y_train, y_test, X, Y, models):
    for x in models:
        print('{}'.format(x))
        model = x
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print('Confusion Matrix :')
        print(confusion_matrix(y_test, predictions))
        print('Accuracy Score :', accuracy_score(y_test, predictions))
        print('Report : ')
        print(classification_report(y_test, predictions))

        kfold = KFold(n_splits=10, random_state=7)

        print(cross_val_predict(model, X, Y, cv=kfold))

        result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
        print(result)
        print("Accuracy: %.3f%% (%.3f%%)" % (result.mean() * 100.0, result.std() * 100.0))

def plot(data):
    pd.crosstab(data.cp, data.target).plot(kind ="bar")
    plt.title('Heart Disease Frequency According To CP')
    plt.xlabel('CP')
    plt.xticks(rotation=0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency of Disease or Not')

    pd.crosstab(data.fbs, data.target).plot(kind="bar")
    plt.title('fbs')
    plt.xlabel('fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
    plt.xticks(rotation=0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency of Disease or Not')

    pd.crosstab(data.exang, data.target).plot(kind="bar")
    plt.title('exercise induced angina')
    plt.xlabel('exercise induced angina (1 = yes; 0 = no)')
    plt.xticks(rotation=0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency of Disease or Not')

    pd.crosstab(data.slope, data.target).plot(kind="bar")
    plt.title('slope of the peak exercise ST')
    plt.xlabel('the slope of the peak exercise ST segment')
    plt.xticks(rotation=0)
    plt.legend(["Haven't Disease", "Have Disease"])
    plt.ylabel('Frequency of Disease or Not')
    plt.show()

    plt.scatter(x=data.age[data.target == 1], y=data.thalach[data.target == 1], c='red')
    plt.scatter(x=data.age[data.target == 0], y=data.thalach[data.target ==0], c ='green')
    plt.title('thalach')
    plt.xlabel('age')
    plt.xticks(rotation=0)
    plt.legend(["Have Disease", "Haven't Disease"])
    plt.ylabel('heart rate')
    plt.show()

    plt.scatter(x=data.chol[data.target == 1], y=data.thalach[data.target == 1], c='red')
    plt.scatter(x=data.chol[data.target == 0], y=data.thalach[data.target == 0], c='green')
    plt.title('thalach / chol')
    plt.xlabel('max heart rate')
    plt.xticks(rotation=0)
    plt.legend(["Have Disease", "Haven't Disease"])
    plt.ylabel('chol (mg/dl)')
    plt.show()




if __name__ == "__main__":
    data, male, female = get_data(DATA_DIR)
    

    x_train, x_test, y_train, y_test, X, Y = create_sets(data)
    train(x_train, x_test, y_train, y_test, X,Y, models)

    # plot(data)