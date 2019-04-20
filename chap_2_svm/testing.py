from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import  pandas as pd

ctx= '../data/'
csv=pd.read_csv(ctx+"bmi.csv")
label =csv['label']
w=csv['weight']/100#최대 100kg
h=csv['height']/200#최대 2m

wh=pd.concat([w,h],axis=1)#w+h
#학습데이터와 테스트데이터분리
data_train,data_test,label_train,label_test=train_test_split(wh,label)
clf=svm.SVC()
clf.fit(data_train,label_train)
predict=clf.predict(data_test)

ac_score=metrics.accuracy_score(label_test,predict)
cl_report=metrics.classification_report(label_test,predict)
print("정답률:",ac_score)
print("리포트:\n" ,cl_report)

'''

 "avoid this warning.", FutureWarning)
정답률: 0.9894
리포트:
               precision    recall  f1-score   support

        fat        1.00      1.00      1.00      2671
     normal        0.97      0.99      0.98      1215
       thin        1.00      0.98      0.99      1114

   micro avg       0.99      0.99      0.99      5000
   macro avg       0.99      0.99      0.99      5000
weighted avg       0.99      0.99      0.99      5000


Process finished with exit code 0

'''