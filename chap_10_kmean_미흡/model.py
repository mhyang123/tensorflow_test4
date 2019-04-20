'''
k-평균
군집화(clistring)문제를 풀기위한 자율 학습 알고리즘
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

num_point=2000
vectors_set=[]

for i in range(num_point):
    if np.random.random()>0.5:
        vectors_set.append([np.random.normal(0.0,0.9),
                            np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0,0.5),
                            np.random.normal(1.0,0.5)])

df = pd.DataFrame({

'x':[v[0]for v in vercors_set],
'y':[v[1]for v in vercors_set]
})

sns.lmplot('x','y')