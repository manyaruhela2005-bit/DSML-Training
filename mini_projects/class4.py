# 31-aug-26

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(pd.set_option("display.max_columns", None))# display all the possibe columns
print(pd.set_option("display.float_format", lambda x: f"{x:,.2f}"))#x:-> formaitting Nos.- 12345678.0123....

np.random.seed(42)

#CREATE THE DATASET
n=120

age= np.random.randit(18,61,n)
income=np.random.normal(55000,15000,n).clip(20000,120000)
purchases=np.random.poisson(5,n)+1
spending=(income* np.random.uniform(0.02,0.10,n)+
          purchases* np.random.uniform(150,500,n))
satisfaction=np.random.normal(7,1.2,n).clip(1,10)

