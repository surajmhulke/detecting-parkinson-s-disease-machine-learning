# Detecting Parkinson's Disease using Machine Learning

Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms may not be noticeable, but signs of stiffening, tremors, and slowing of movements may be indicative of Parkinson's disease.

But there is no definitive way to determine whether a person has Parkinson's disease because there are no specific diagnostic methods available. What if we use machine learning to predict whether a person suffers from Parkinson's disease or not? This is exactly what we will explore in this project.

## Parkinson's Disease Prediction using Machine Learning in Python

### Importing Libraries and Dataset

Python libraries make it very easy for us to handle data and perform both typical and complex tasks with just a few lines of code. We'll be using the following libraries:

- **Pandas**: This library helps us load data into a DataFrame in a 2D array format and has multiple functions to perform analysis tasks.
- **Numpy**: Numpy arrays are very fast and can perform large computations quickly.
- **Matplotlib/Seaborn**: These libraries are used for data visualization.
- **Scikit-learn (Sklearn)**: This module contains multiple libraries with pre-implemented functions for tasks ranging from data preprocessing to model development and evaluation.
- **XGBoost**: This library includes the eXtreme Gradient Boosting machine learning algorithm, which helps achieve high prediction accuracy.
- **Imblearn**: This module contains functions for handling problems related to data imbalance.

You can install the necessary packages using the following command (install first time only):

```python
!pip install numpy pandas sklearn xgboost --upgrade

Next, we'll import these libraries and the dataset:

python

import numpy as np
import pandas as pd

# Input data files are available in the read-only "/kaggle/input/" directory
# Use the appropriate path for your dataset

# Read the dataset into a DataFrame
df = pd.read_csv('/kaggle/input/parkinsons.data')

Data Collection

We'll start by reading the dataset into a DataFrame and displaying the last 5 rows using df.tail(). You can use df.head() to see the first 5 rows. The dataset should provide information on various features and whether the individual has Parkinson's disease.
Feature Selection

To prepare our data for modeling, we need to select the relevant features. We'll use the chi-square test to reduce the feature space to 30, as having too many features can lead to overfitting. Here's how we'll do it:

python

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Get all features except 'status'
features = df.loc[:, df.columns != 'status'].values[:, 1:]

# Get status values in array format
labels = df.loc[:, 'status'].values

# Normalize features
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)
y = labels

# Use chi-square test for feature selection
selector = SelectKBest(chi2, k=30)
selector.fit(X, df['status'])
filtered_columns = selector.get_support()
filtered_data = X[:, filtered_columns]

# Add the 'status' column back
filtered_data = np.column_stack((filtered_data, labels))

# Update the DataFrame
df = pd.DataFrame(data=filtered_data, columns=[df.columns[i] for i, feature in enumerate(filtered_columns) if feature] + ['status'])

Model Training

We'll use the XGBoost classifier for model training. Here's how you can do it:

python

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Initialize and train the XGBoost model
model = XGBClassifier()
model.fit(x_train, y_train)

Model Prediction

We can make predictions using our trained model and evaluate its accuracy:

python

# Make predictions
y_prediction = model.predict(x_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_prediction) * 100
print("Accuracy Score is", accuracy)

```
  Finnaly pridict the model

y_prediction = model.predict(x_test)

print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)
Accuracy Score is 90.0

Summary
In this Python machine learning project, we learned to detect the presence of Parkinson’s Disease in individuals using various factors. We used an XGBClassifier for this and made use of the sklearn library to prepare the dataset. This gives us an accuracy of 96.66%, which is great considering the number of lines of code in this python project.
