"""date: 1.oct.26
SVM- support vector machine
•Supervised learning algorithm
•Used for classification and regression problems
•SVM is a discriminative classifier formally defined by a separating hyperplane
•separates classes with maximum possible margin
•focuses on support vectors to build the hyperplane
•SVM can be used for linear and non-linear classification problems
"""
"""Key Idea-
-finds hyperplane that
-separates classes with maximum margin
-determined by the support vectors
-svm finds the boundary that is farthest from both the classes"""
"""Types-
1. Linear SVM
•Used when data is linearly separable
•Finds the hyperplane that separates the classes with maximum margin
•Support vectors are the data points that are closest to the hyperplane

2. Non-Linear SVM
•Used when data is not linearly separable
•Uses kernel trick to transform the data into higher dimensions
•Finds the hyperplane that separates the classes with maximum margin in the transformed space"""

"""USE:
1. handles complex data NL
2. Good Generalization
3. Works well with high dimensional data
4. small or medium sized datasets
5. works well with clear margin of separation
6. effective in high dimensional spaces
7. memory efficient
8. versatile- different kernel functions can be specified for the decision function"""

"""working
1. understand the problem and collect data
2. visualize the data
3. undersyand the SVR model
4. choose the kernel function
5. train the model
6. evaluate the model
7. tune the hyperparameters
8. make predictions on new data"""

#laptop buying prediction using SVM
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC


path="/Users/manyaruhela/Desktop/laptop_buying_data.csv"

df=pd.read_csv(path, na_values=["", " ", "NA", "N/A", "null"," NULL", "none", "?"])#handle missing values

#data inspection
print("shape:", df.shape)
print("\nmissing values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

df= df.drop_duplicates()

X= df.drop("Buys Laptop", axis=1)#i/p features  
y= df["Buys Laptop"]#o/p feature

numeric_feautres = ["Age", "Credit Score"]# list of numeric features
categorical_features = ["Income", "Student", "Location"]# list of categorical features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)# statify balances the classes in train and test sets 

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # handle missing values
    ("scaler", StandardScaler()) # scale the numeric features
    ])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # handle missing values
    ("onehot", OneHotEncoder(handle_unknown="ignore")) # encode categorical features
    ])
preprocessor_pipeline = ColumnTransformer([
    ("num", numeric_pipeline, numeric_feautres) #apply numeric pipeline to numeric features
    ("cat", categorical_pipeline, categorical_features) #apply categorical pipeline to categorical features
    ])  

