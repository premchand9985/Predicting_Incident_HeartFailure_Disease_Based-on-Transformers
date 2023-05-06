# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use("ggplot")
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read the csv data
file_path = "heart.csv"
data_df = pd.read_csv(file_path)
data_df.head()
data_df.shape
data_df.info()
data_df.describe()
# Plot histograms for each variable
data_df.hist(figsize=(12, 12))
plt.show()

pd.crosstab(data_df.age, data_df.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(data_df.corr(), annot=True, fmt='.1f')
plt.show()

data_df.isnull().any()

data_df = data_df.dropna(axis=0)

# Transform data to numeric to enable further analysis
data_df = data_df.apply(pd.to_numeric)
data_df.dtypes

data_df.describe()
X_data = np.array(data_df.drop(['target'], 1))
y_data = np.array(data_df['target'])
X_data[0]
mean = X_data.mean(axis=0)
X_data -= mean
std = X_data.std(axis=0)
X_data /= std
X_data[0]
# Create X and Y datasets for training
from sklearn import model_selection

X_train_data, X_test_data, y_train_data, y_test_data = model_selection.train_test_split(X_data, y_data, stratify=y_data, random_state=42, test_size=0.2)
# Convert the data to categorical labels
from keras.utils.np_utils import to_categorical

Y_train_data = to_categorical(y_train_data, num_classes=None)
Y_test_data = to_categorical(y_test_data, num_classes=None)
print(Y_train_data.shape)
print(Y_train_data[:10])
X_train_data[0]

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# Define a function to build the Keras model
def build_model():
    # Create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

heart_model = build_model()

print(heart_model.summary())

# Fit the model to the training data
training_history = heart_model.fit(X_train_data, Y_train_data, validation_data=(X_test_data, Y_test_data), epochs=50, batch_size=10)

from sklearn.metrics import classification_report, accuracy_score
categorical_predictions = np.argmax(heart_model.predict(X_test_data), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test_data, categorical_predictions))
print(classification_report(y_test_data, categorical_predictions))
