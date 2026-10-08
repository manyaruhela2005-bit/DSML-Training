"""
Random Forest
step 1: Import the necessary libraries
step 2: Load the dataset
step 3: Preprocess the data
step 4: Split the dataset into training and testing sets
step 5: Train the Random Forest model
step 6: Make predictions on the test set
step 7: Evaluate the model's performance
step 8: Visualize the results
"""
#predict laptop buying occurances 
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score #cross validation- improves model performance
from sklearn.compose import ColumnTransformer # preprocessing 
from sklearn.pipeline import Pipeline #automate steps
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer #encoding categorical variables, simple imputer for missing values
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load and upload the dataset
path="/Users/manyaruhela/Downloads/laptop_buying_data.csv"
df=pd.read_csv(path, na_values=["", " ", "NA", "N/A", "null"," NULL", "none", "?"])#handle missing values
print(df.head())
print("--------------------------------")

 #data inspection
# rows and columns
print(df.shape)
print("--------------------------------")
#columns names
print(df.columns.tolist())
print("--------------------------------")
#dataset information
print(df.info())
print("--------------------------------")
#checking for missing values
print(df.isnull().sum())
print("--------------------------------")
# #drop missing rows
# df= df.dropna()
# print("--------------------------------")
# #check again
# print(df.isnull().sum())
# print("--------------------------------")
# #drop duplicates
# df = df.drop_duplicates()

#statistical summary of the dataset
print(df.describe(include="all"))
print("--------------------------------")

#separate X input features and y output
X= df.drop("Buys Laptop", axis=1)#all except buys laptop
y= df["Buys Laptop"]#target

numeric_features = ["Age", "Credit Score"]
categorical_features = ["Income", "Student", "Location"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #straitify balances the classes in train and test sets (Normalization)
    )

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))#handle missing values
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), #handle missing values
    ("onehot", OneHotEncoder(handle_unknown="ignore")) #encoding categorical variables
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])  

#model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))
])

#train the model
model.fit(X_train, y_train) 

#make predictions
y_pred = model.predict(X_test) 
print("Predictions:", y_pred)

#accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 2))