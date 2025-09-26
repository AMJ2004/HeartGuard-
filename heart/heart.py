import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings


warnings.filterwarnings('ignore')
df = pd.read_csv('heart.csv')
df=df.drop(['oldpeak', 'slope','ca', 'thal'], axis=1)

#print(df.head())
#print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()

#print(df.head())
X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)
print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1 =model1.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, prediction1)
print(cm)
sns.heatmap(cm, annot=True,cmap='BuPu')
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print(TP,TN,FN,FP)
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))
with open('lr_pickle','wb') as f:
    pickle.dump(lr,f)
with open('lr_pickle','rb') as f:
   mp= pickle.load(f)



