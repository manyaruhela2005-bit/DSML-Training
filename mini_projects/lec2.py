"""Pandas"""
#series and dataframes
#series is used for one dimensional data and dataframe is used for two dimensional data
#excel built in python is pandas    
import pandas as pd

#dictoonary
# data = {
#     'Name': ['Manya', 'Utkarsh','Sargam', 'Surabhi'],
#     'Age': [21, 19, 23, 23]
# }
#creating a dataframe
# df = pd.DataFrame(data)
# #to display the first 5 rows
# df.head()

#read csv format
df=pd.read_csv("/Users/manyaruhela/Desktop/student_performance_prediction.csv")
df.info()
df.describe()

#selecting a column
#print(df['Student ID'].head(10))
#print(df['Student ID'].tail(10))

#selection of rows by index
print(df.loc[0])
#selecting rows from 0 to 5
print(df.loc[0:5]) 