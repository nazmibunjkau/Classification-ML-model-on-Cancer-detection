# Author: Nazmi Bunjaku
# CIS 492 - Big Data Analytics
# Professor Sunnie Chung
# Lab 5 - Classification with Machine Learning
# April 18, 2023

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, cross_val_score

# Load the dataset into a pandas dataframe
data = pd.read_csv("breast_cancer_dataset.csv")

# Handle missing values
data = data.dropna()

# Handle outliers
# Split the data into feature set (X) and target variable (y)
X = data.drop(["Classification"], axis=1)
y = data["Classification"]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the SVM classifier
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Calculate the accuracy, recall, precision, and macro F1 scores of the classifier using a confusion matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=1)
precision = precision_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Print results
print("Confusion Matrix:")
print(cm)
print("Accuracy: ", acc)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)

# Implement a 5-fold cross validation (k=5) to compare the accuracy of each test of the classifier
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train the SVM classifier and calculate the accuracy of the classifier for each fold
svm = SVC(kernel='linear', C=1)
scores = cross_val_score(svm, X, y, cv=kf)

# Compute the overall accuracy by taking the average of each accuracy of each fold
overall_acc = scores.mean()

# Print results
print("Overall Accuracy: ", overall_acc)

# Use the trained classifier to make predictions on a new test dataset
svm_new = SVC(kernel='linear', C=1)
svm_new.fit(X, y)
svm.fit(X, y)
y_new_pred = svm.predict(X_test)

# Calculate the accuracy, recall, precision, and macro F1 score of the classifier using the confusion matrix
new_cm = confusion_matrix(y_test, y_new_pred)
new_acc = accuracy_score(y_test, y_new_pred)
new_recall = recall_score(y_test, y_new_pred, pos_label=1)
new_precision = precision_score(y_test, y_new_pred, pos_label=1)
new_f1 = f1_score(y_test, y_new_pred, pos_label=1)

# Display the confusion matrix, accuracy, recall, precision, and macro F1 score
print("\nConfusion Matrix:")
print(new_cm)
print("Accuracy:", new_acc)
print("Recall:", new_recall)
print("Precision:", new_precision)
print("F1 Score:", new_f1)
