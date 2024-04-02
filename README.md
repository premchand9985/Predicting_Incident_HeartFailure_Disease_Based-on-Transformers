# HeartFailure_Disease_Based-on-Transformers

This project aims to predict the presence of heart disease based on various medical factors. The dataset used in this project contains several attributes such as age, sex, cholesterol levels, and exercise-induced angina, among others, which are used to train machine learning models.

## Dataset
The dataset used in this project is named "heart.csv". It contains the following columns:

Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise induced angina
ST depression induced by exercise relative to rest
Slope of the peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thal

## Getting Started
To run this project locally, follow these steps:

Clone this repository to your local machine.

Install the required dependencies using the following command:

pip install -r requirements.txt

Execute the code in a Python environment. The main script is named heart_disease_prediction.py.

## Exploratory Data Analysis (EDA)
The script starts with loading the dataset and performing exploratory data analysis (EDA) tasks such as displaying the first few rows, statistical summary, and information about the dataset.

Visualization techniques such as count plots and bar plots are used to understand the distribution and relationships between variables.

### Model Building and Evaluation
- Several machine learning algorithms are used to build predictive models including:
Logistic Regression
Naive Bayes
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
XGBoost
Neural Network
- Each model is trained on the training set and evaluated on the test set. Accuracy scores are computed and printed out.

## Results
- The accuracy scores of different algorithms are plotted in a bar chart to visualize the performance of each model.

## Contributors
Prem Chand Koru
