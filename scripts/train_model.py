# -------------------------------------------------------------
# Import necessary libraries
# pandas → used for reading and handling the dataset
# scikit-learn → used for machine learning tasks
# seaborn & matplotlib → used for visualization (confusion matrix)
# -------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib
import matplotlib.pyplot as plt



# -------------------------------------------------------------
# Load the dataset containing extracted audio features
# This dataset was created after feature extraction and merging
# -------------------------------------------------------------

df = pd.read_csv("../features/features_all.csv")

print("Dataset shape:", df.shape)
print("\nFirst few rows of dataset:")
print(df.head())


# -------------------------------------------------------------
# Separate the input features (X) and the target label (y)
# X contains all numerical acoustic features
# y contains the category we want the model to predict
# -------------------------------------------------------------

X = df.drop(columns=["label", "category"])
y = df["category"]


# -------------------------------------------------------------
# Split the dataset into training and testing sets
# 80% of the data will be used to train the model
# 20% will be used to test the model's performance
# random_state ensures the split is reproducible
# -------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------------------
# Create the machine learning model
# RandomForestClassifier works well for tabular data
# n_estimators = number of decision trees
# -------------------------------------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)


# -------------------------------------------------------------
# Train the model using the training dataset
# The model learns patterns from the acoustic features
# -------------------------------------------------------------

model.fit(X_train, y_train)


# -------------------------------------------------------------
# Use the trained model to predict categories for the test data
# -------------------------------------------------------------

y_pred = model.predict(X_test)


# -------------------------------------------------------------
# Calculate the accuracy of the model
# Accuracy = number of correct predictions / total predictions
# -------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# -------------------------------------------------------------
# Display a detailed classification report
# Shows precision, recall, and F1-score for each class
# -------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------------------
# Generate and display the confusion matrix
# This visualizes correct vs incorrect predictions
# -------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# -------------------------------------------------------------
# Save the trained model so it can be reused later
# -------------------------------------------------------------

joblib.dump(model, "../sound_classifier.pkl")

print("Model saved as sound_classifier.pkl")