#import libraries
import pandas as pd #create dataframes, read csv files, data cleaning, data manipulation
import matplotlib.pyplot as plt #visualization of data

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree #algorithm and plot_tree-> to visualize the decision tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #r2 -> evaluation of the model

#load the dataset
df= pd.read_csv("/Users/manyaruhela/Downloads/house_prices.csv")
print(df.head()) #display the first 5 rows of the dataset

#create features
X=df[["Size", "Bedrooms", "Age"]]
y=df["Price"]

#split data
#X-input, y- output, random_state-> reproducibility of the results
X_train, X_test, y_train, y_test= train_test_split( X, y, test_size=0.2, random_state=42) 

#create and train the model
model= DecisionTreeRegressor(max_depth=3,
                             random_state=42)

model.fit(X_train, y_train)

#visualize the decision tree
plt.figure(figsize=(15,9))

plot_tree(model, feature_names=X.columns, filled=True, rounded=True)

plt.show()

#make predictions 

y_pred= model.predict(X_test)
results=pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(results)
