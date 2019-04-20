import pandas as pd  # built-in package no need to use "from"  as is for aliasing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # show muliple graphs

ctx = 'C:/Users/ezen/PycharmProjects/day_1/titanic/data/'
# ctx means contents
train = pd.read_csv(ctx + 'train.csv')
test = pd.read_csv(ctx + 'test.csv')
# df = pd.DataFrame(train)
# print(df.columns)

'''
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
columns of the titanic data
PassengerId
Survival    Survival    0 = No, 1 = Yes
Pclass    Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Sex    Sex    
Age    Age in years    
SibSp    # of siblings / spouses aboard the Titanic    
Parch    # of parents / children aboard the Titanic    
Ticket    Ticket number    
Fare    Passenger fare    
Cabin    Cabin number    
Embarked    Port of Embarkation    C = Cherbourg, Q = Queenstown, S = Southampton
'''
'''
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Survived'][train['Sex'] == 'male'].value_counts().plot.pie(explode=[0,0.1],
                                          autopct="%1.1f%%", ax=ax[0], shadow=True)
train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0,0.1],
                                          autopct="%1.1f%%", ax=ax[1], shadow=True)
ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(Female)')
'''

# sns.countplot('Survived', data=train, ax=ax[1])
# ax[1].set_title('Survived')
# plt.show()
# probability of male's survive 18.9%   death rate  81.2%
# probability of female's survive rate 74.25%   death rate 25.8%

# pClass

df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1, df_2, margins=True)
# print(df.head())

'''
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
'''
'''
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('No of Passengers Pclass')
sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('Male - Female Pclass')
sns.countplot('Embarked', hue= 'Survived', data=train, ax=ax[1,0])
ax[1,0].set_title('Pclass vs. Survived')
sns.countplot('Pclass', data=train, ax=ax[1,1])
ax[1,1].set_title('Embarked vs. Pclass')
plt.show()
'''
'''
위 데이터를 보면 절반 이상의 승객이 ‘Southampton’에서 배를 탔으며, 
여기에서 탑승한 승객의 70% 가량이 남성이었습니다. 
현재까지 검토한 내용으로는 남성의 사망률이 여성보다 훨씬 높았기에 
자연스럽게 ‘Southampton’에서 탑승한 승객의 사망률이 높게 나왔습니다.
또한 ‘Cherbourg’에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 
높은 것으로 보아서 이 동네는 부자동네라는 것을 예상할 수 있습니다.
'''
# 결측치  - 빠져있는 데이타.  e.g 재산 넣으라고 하면 안 넣는 경우나, 어떤 이유에서 빠진 정보
# 결측치 제거
print(train.isnull().sum())

'''
data is for training (train.csv), data for test (test.csv)
'''


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()

# bar_chart('Sex')
# bar_chart('Pclass')
# bar_chart('SibSp')
# bar_chart('Parch')
# bar_chart('Embarked')

#cabin,TICKET값 삭제

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)

train.head()

test.head()

#train=train.drop(['Cabin']),axis=1)
#test=test.drop(['Cabin']),axis=1)
#train=train.drop(['Ticket']),axis=1)
#test=test.drop(['Ticket']),axis=1)
#train.head()
#test.head() 확인 하고 삭제 한다

#Embarked 값 가공
"""
s_city=train[train['Embarked ']=='S'].shape[0]#Embarked행에서 값이 s인 값을 뽑는다.0차원 =스칼라 값을 만들어라.
c_city=train[train['Embarked ']=='C'].shape[0]
q_city=train[train['Embarked ']=='Q'].shape[0]
#print("S={},C={},Q={})".FORMAT(s_city,c_city,q_city))
print("S=",s_city) #644
print("C=",c_city)#168
print("Q=",q_city)#77
"""

s_city = train[train["Embarked"]=='S'].shape[0]

print("S :", s_city) # S : 646

c_city = train[train["Embarked"]=='C'].shape[0]

print("C :", c_city) # C : 168

q_city = train[train["Embarked"]=='Q'].shape[0]

print("Q :", q_city) # Q : 77



city_mapping = {"S":1, "C":2, "Q":3}

train['Embarked'] = train['Embarked'].map(city_mapping)

test['Embarked'] = test['Embarked'].map(city_mapping)

train.head()

test.head()
"""
train=train.fillna({"Embarked":"S"})
city_mapping={"S":1,"C":2,"Q":3}
train[train['Embarked ']=train[train['Embarked '].map(city_mapping)
test[train['Embarked ']=test[train['Embarked '].map(city_mapping)
#print(train.head())
#print(test.head())
"""
#================

combine = [train, test]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train['Title'], train['Sex']))

#https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C_%ED%91%9C%ED%98%84%EC%8B%9D

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
train[['Title','Survived']].groupby(['Title'], as_index=False).mean()#[[]]행열표시 ,groupby구룹핑하기
#print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())#결과를 보기 ;보고 #으로 막기
"""
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.285714
5   Royal  1.000000
"""
train = train.drop(['Name','PassengerId'], axis = 1)

test = test.drop(['Name','PassengerId'], axis = 1)

combine = [train, test]

train.head()
#print(train.head())
#----------------------------------
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0) # fillna

print(train.head())
#----------------------------------------
sex_mapping = {"male":0, "female":1}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train.head()
#-------------------------------------
#AGE 가공하기
train['Age'] = train['Age'].fillna(-0.5)# -1 0사이에서는 Unknown으로 처리함

test['Age'] = test['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)#나이를 잘라서 담아라

test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)#나이를 잘라서 담아라:편집한다


#print(train.head())
train.head()
#------------------------------------
#mapping시작//이부분 코드 수정 필요 (먼저 age_mapping하고age_title_mapping진행 하여야 함

age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = train['AgeGroup'].map(age_mapping)

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)

print(train.head())

age_title_mapping = {0:"Unknown",1: "Young Adult", 2:"Student",
                     3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"} #0:"Unknown"코드추가 됨 (네이버 코드랑 다름)

for x in range(len(train['AgeGroup'])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]

print(train.head())

#----------------------
#fare처리
train["FareBand"]=pd.qcut(train['Fare'],4,labels={1,2,3,4,})
test["FareBand"]=pd.qcut(test['Fare'],4,labels={1,2,3,4,})


train = train.drop('Fare', axis = 1)
test = test.drop('Fare', axis = 1)

#------------------
#데이터 모델링
#-----------------


train_data = train.drop('Survived', axis = 1)

target = train['Survived']

print(train_data.shape)
print(target.shape)
print(train.info)