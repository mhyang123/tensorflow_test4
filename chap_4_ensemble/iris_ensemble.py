from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import  pandas as pd
import numpy as np

np.random.seed(0)#랜덤값을 고정시키는 값
iris = load_iris ()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
print(df.columns)
'''
Index(['sepal length (cm)', 
        'sepal width (cm)',
        'petal length (cm)',
       'petal width (cm)'],
 '''
df["species"]=pd.Categorical.from_codes()
dfp['is_train'=np.random.u (0,1,len(df))<=.75]#75%를 학습시킨다



#****
#러닝
#***

clt=RandomForestClassifier(n_jobs=2,range_)


#****
#testing
#***

preds=iris.targe_name[clf.prdict(test[featu])]



















