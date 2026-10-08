8-oct-26
"""Gradient boosting-XGBoost and LightGBM"""
# what is gradient boosting?
"""1. ensemble learning technique building multiple decision trees
2. each new tree- tries to correct the errors made by prev trees
3. combines all trees to make a final stronger prediction"""

#How Gradient Boosting works
""".1 start with simple model:- begin with an initial prediction
2. Calculate the error:- Find the diff b/w  
3. train new tree
4. update predictions
5. repeat
6. final output"""

#XGBoost Vs LightGBM-> used for categorical data
"""1. XGBoost (Extreme Gradient Boosting)
- Builds decision trees sequentially, correcting previous errors.
- Uses regularization (L1/L2) to reduce overfitting.
- Uses level-wise tree growth.
- Known for high accuracy and robustness.

2. LightGBM (Light Gradient Boosting Machine)
- Developed by Microsoft.
- Uses leaf-wise tree growth.
- Faster training and lower memory usage, especially on large datasets.
- Can overfit on small datasets if not tuned properly."""

'''========================CODE========================='''
#import reqlibraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

#load the dataset
df=pd.read_csv("/Users/manyaruhela/Downloads/banking_loan_default_dataset_500_rows.csv")

print(df.head)
print("------------------------------------------")
print(df.shape)
print("------------------------------------------")
print(df.info)
x=df.isnull().sum().sort_values(ascending=False).head(15)#top 15
print("------------------------------------------")
print(x)

targe_column='loan_default'
df=df.dropna(subset=[targe_column]).copy()

X=df.drop(columns=[targe_column])
y=df[targe_column]
print("------------------------------------------")
print('Target values: ')
print(y.value_counts())

label_encoder= LabelEncoder()
y=label_encoder.fit_transform(y.astype(str))
print("------------------------------------------")
print('classes: ', list(label_encoder.classes_))
print('Encoded classes: ', np.unique(y))

numeric_features=X.select_dtypes(include=np.number).columns.tolist()
categorical_features=X.select_dtypes(exclude=np.number).columns.tolist()
print("------------------------------------------")

print('Numerical features:', numeric_features)
print('Categorical features:', categorical_features)
