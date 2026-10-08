"""EDA using Salary dataset"""
import numpy as np          
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/Users/manyaruhela/Downloads/salary_dataset (1) (1).csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#UNIVARIATE ANALYSIS
#Distribution of emplooyees b y salary
plt.figure(figsize=(8,6))

#kde=True is used to plot the kernel density estimation which is a smoothed version of the histogram
#bins=10 is used to specify the number of bins in the histogram
sns.histplot(df['Salary'], kde=True, bins=10)
plt.title('Distribution of Salary')
plt.xlabel('Salary')
plt.ylabel('number of employees')   
#plt.show()

#distribution of employees by experience
sns.histplot(df['Experience'], kde=True, bins=10)
plt.title('experience distribution')
plt.xlabel('Experience(years)')
plt.ylabel('number of employees')   
#plt.show()

#number of employees by job role
plt.figure(figsize=(10,5))

sns.countplot(data=df, x='JobRole')

plt.title('Number of Employees by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)

#plt.show()

#bivariate analysis
#salary and JobRole
plt.figure(figsize=(10,5))

sns.barplot(data=df, x='JobRole', y='Salary')   

plt.title('Salary by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Salary')
plt.xticks(rotation=45)

#plt.show()

#salary by education
plt.figure(figsize=(8,5))

sns.barplot(data=df, x='Education', y='Salary')

plt.title('Salary by Education')
plt.xlabel('Education')
plt.ylabel('Average Salary')
#plt.show()

#salary by location
plt.figure(figsize=(8,5))

sns.barplot(data=df, x='Location', y='Salary')

plt.title('Salary by Location')
plt.xlabel('Location')
plt.ylabel('Average Salary')
#plt.show()

#with regression line
plt.figure(figsize=(8,5))
sns.regplot(data=df, x='Experience', y='Salary')
plt.title('Salary by Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')    
plt.show()