"""Numpy"""
import numpy as np
lst=[1, 2, 3, 4,5 ]
arr=np.array(lst)
print(lst*2)
print(arr*2)
td=np.array([[1, 2], [3, 4]])
print(td)
print(td[1,1])
print(td[0,1])
print(td[1,0])

print(np.zeros(5)) #o/p-[00000]
print(np.ones(10)) #o/p-[1111111111]

print(np.zeros([2,2])) #o/p-[[0 0]
                       #     [0 0]]

print(np.zeros([3,3])) #o/p-[[0 0 0]
                       #     [0 0 0]
                       #     [0 0 0]]

#useful operations
a=np.array([1,2,3])
b=np.array([4,5,6])
print(a+b) #o/p-[5 7 9]
print(a-b) #o/p-[-3 -3 -3]
print(a*b) #o/p-[4 10 18]
print(a/b) #o/p-[0.25 0.4 0.5]