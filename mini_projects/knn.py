#KNN- no. of nearest neighbors
#non-parametric, instance-based learning algorithm, predicts class or value(regression)  of dp based on majority of k most similar neighbouring data points
"""
pipeline
1. choose k
2. calculate distance between new data point and all training data points
3. find k nearest neighbors based on distance
4. assign class label based on majority vote of k nearest neighbors
5. return predicted class label
"""
#intuition- i am the average of k peaople i am surrounded by
#we are going fo rhigh dimensional data, so we will use euclidean distance to calculate distance between points
"""
    ^
    |
    |      •
    |       \\ distance to be calculated by euclidean distance
    |        •   
    |
    |
    |
    L_____________________>
    """
#how to choose k-
"""1. heuristics- odd number, square root of n, cross validation- simple ruleof thumb, start with underroot n
      Eg- n=400 k=sqrt(400)=20,
       odd number, we can take 19 or 21

2. Experimentation- thy different values of k and see which one gives the best performance using cross validation
    Eg= highest accuracy
    eg
    point- 1,2,3,4
    distance- 2.1, 2.6, 3.1, 3.2
    class- 0, 0, 1, 1

2. elbow method- plot accuracy vs k, choose k at elbow point 
3. domain knowledge- if we know the data is noisy, we can choose a larger k
4. grid search- try different values of k and choose the one that gives the best performance"""

#decision Boundry
""" it is a line or curve that separates different classes in the feature space. In KNN, the decision boundary is determined by the majority class of the k nearest neighbors. The decision boundary can be linear or non-linear
- linear decision boundary- when the classes are linearly separable, the decision boundary will be a straight line. For example, if we have two classes, A and B, and the data points of class A are on one side of the line and the data points of class B are on

how do we get the decision boundary- 
1. generate a grid of points in the feature space
2. use trained model rg-knn on the actual trainin g data
3. predict the class (0,1) for each point
4. colour each grid point based on predicted class
5. the decision boundary appears as the line or curve that separates the different colored regions in the feature space"""


#OVER-FITTING AND UNDER-FITTING IN KNN
"""
1. k is Very small- model will be very sensitive to small changes in the training data, decision boundary will be very complex and irregular(follows each training point), learns noise and outliers, high variance, low bias, overfitting
2. K is intermediate- Decision bondary will be smoother and more generalizable, ignores small noise and outliers, low variance, low bias, good fit, performs well on unseen data
3. K is very large- for a new point, model looks at almost all training points, prediction is based on majority class of training data, decision boundary will be very smooth and simple, results in underfitting, do not capture the actual pattern in ds, fails to learn complex relationships, high bias, low variance, poor performance on unseen data
"""

#failure cases of knn
'''
1. Large Dataset-lazy learning algorithm(no training, stores all data), for a query point, it needs to calculate distance to all training points, then it starts the distance and picks the k nearest neighbors to find the majority class
2. High Dimensional Data- curse of dimensionality, as the number of features increases, the distance between points becomes less meaningful, all points become equidistant, model struggles to find nearest neighbors, performance degrades
3. Imbalanced Dataset- if one class is significantly more frequent than the other, the model will be biased towards the majority class, leading to poor performance on the minority class, accuracy may be misleading, precision, recall, f1-score should be used to evaluate performance
4. outliers- presence of outliers can significantly affect the performance of KNN, as they can skew the distance calculations and lead to incorrect predictions. Outliers can be removed or handled using techniques such as robust scaling or outlier detection methods.
5.non-homogeneous data- when the data points are not uniformly distributed across the feature space, the model may struggle to find meaningful neighbors, leading to poor performance
6.imbalanced data- when one cclass has more samples than the other, the model will be biased towards the majority class, leading to poor performance on the minority class, accuracy may be misleading, precision, recall, f1-score should be used to evaluate performance
7. inference and not for prediction- KNN is a lazy learning algorithm, it does not learn a model from the training data, it simply stores the training data and uses it to make predictions for new data points. This means that KNN is not suitable for making predictions on new data points that are significantly different from the training data, as it will not have any information about those points.
    KNN is better suited for inference tasks, where we want to understand the relationships between the features and the target variable in the training data.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df= pd.read_csv('/Users/manyaruhela/Desktop/knn_customer_dataset_240_rows.csv')
print("------------------------------------------------")
print("head:")
print(df.head())
print("------------------------------------------------")
print("shape:")
print(df.shape)
print("------------------------------------------------")
print("info:")
print(df.info())
print("------------------------------------------------")
print("describe:")
print(df.describe())

#change this to the actual target column in ds
target_column = 'target'  # Replace with the actual target column name

x=df.drop(columns=[target_column])#i/p
y=df[target_column] #o/p

x.head()
y.head()
y.value_counts()#proportion of each class inside a target, y=taret column

x=x.select_dtypes(include=np.number)#selecting only numerical columns, dropping categorical columns

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)#stratify=y ensures that the class distribution in the train and test sets is similar to the original dataset
print("------------------------------------------------")
print("training data: ", x_train.shape)
print("testing data: ", x_test.shape)
print("------------------------------------------------")
print("training")
scalar=StandardScaler()#standardize the data, mean=0, std=1

x_train_scaled=scalar.fit_transform(x_train)# actual training data-<std deviation,avg>-based in training data- scale
x_test_scaled=scalar.transform(x_test) #it already has the mean and std from training data, so we use transform, not fit_transform

#design the model
knn=KNeighborsClassifier(n_neighbors=5)#k=5, can be changed to any odd number, can be tuned using grid search or cross validation
knn.fit(x_train_scaled, y_train)

#predict on test data
y_pred=knn.predict(x_test_scaled)

print("------------------------------------------------")
print("accuracy score: ", accuracy_score(y_test, y_pred))
print("------------------------------------------------")
print("classification report: ")
print(classification_report(y_test, y_pred))
print("------------------------------------------------")
print("confusion matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("------------------------------------------------")

sns=sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')#- colour mapping
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("Confusion Matrix")
plt.show()