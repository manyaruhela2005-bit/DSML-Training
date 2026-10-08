#all 1: create a population 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#imagine these are spending amounts of 1000 customers

pop=np.random.exponential(scale=2000,size=1000)
print("population size:",len(pop))
print("population mean: ₹",round(pop.mean(),2))

plt.list(pop, bin=50)
plt.title("Population Distribution")   
plt.xlabel("Spending Amounts")
plt.ylabel("Frequency")
plt.show()

#see what happens when sample size increases
# sample_sizes=[5,30,100]

# for n in sample_sizes:
#     smp=np.random.choice(pop,n,replace=False)
#     means.append(smp.mean())

#     print(f"Sample size: {n:3d} | Sample mean: ₹{np.mean(means):.2f} | Sample std: ₹{np.std(means):,.2f}")

#=============================================================================================================================================================================================================================================================================================

#hypothesis testing
#its a statistical method where we test a hypothesis about a population parameter based on sample data. we decide if we have enough evidence to reject or reject to fail the hypothesis( null hypothesis)(when their claim is true) or alternative hypothesis(when their claim is false).
#eg- teaching method-> a school claims their average is 75 marks we take 30 students and find their average marks is 70 marks. we can use hypothesis testing to determine if the school's claim is valid or not.
"""5 step process of hypothesis testing:
1. make a claim about the population parameter (null hypothesis) eg-avg score-75
2. set up the hypothesis write H0 and H1(null and alternative hypothesis) eg- H0: avg score=75, H1: avg score!=75
3. collect sample data and calculate the test statistic (eg- t-test, z-test)
4. perform the test and calculate the p-value (probability of observing the sample data if the null hypothesis is true)
5. make a decision based on the p-value """

#three important tets
"""1. t-test: used to compare the means of two groups
2. chi-square test: used to test the association between two categorical variables(counts/frequencies)
3. ANOVA: used to compare the means of three or more groups"""

from scipy import stats

#marks of students in two different teaching methods
method_A=[75,80,85,90,95,70,65,60,55,50]
method_B=[60,65,70,75,80,85,90,95,100,105]  
print("Method A mean:",np.mean(method_A))
print("Method B mean:",np.mean(method_B))

#independent t test
t_stat, p_value=stats.ttest_ind(method_A,method_B)
print("t-statistic:",t_stat)
print("p-value:",p_value)

if p_value<0.05:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")

#product preference by gender
data=[
    [30,40], #male: product A, product B
    [10,40] #female: product A, product B
]

chi2, p_value, dof, expected= stats.chi2_contingency(data)

print("chi- square statistic:", chi2)
print("p-value:", p_value )

if p_value<0.05:
    print("Reject the null hypothesis: There is a significant difference between the three groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the three groups.")

#three teaching methods
method1=[70, 72, 68, 75, 71, 69, 73, 70]
method2=[75,77,74, 76, 78, 75, 79, 76]
method3=[80,82,81,79,83,80,84,82]

#one way anova
f_stat, p_value=stats.f_oneway(method1,method2,method3)
print("F-statistic:", f_stat)
print("p-value:", p_value)

if p_value<0.05:
    print("Reject the null hypothesis: There is a significant difference between the three groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the three groups.")