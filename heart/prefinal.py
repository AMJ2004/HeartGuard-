import pandas as pd
import numpy as np
# Data Resampling
from sklearn.utils import resample
# Data Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Data Splitting
from sklearn.model_selection import train_test_split
# Data Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("framingham.csv")
data["glucose"].fillna((data["glucose"].mode())[0], inplace=True)
data.dropna(inplace=True)

# resampling imbalanced dataset
target1=data[data['TenYearCHD']==1]
target0=data[data['TenYearCHD']==0]

target1=resample(target1,replace=True,n_samples=len(target0),random_state=40)
target=pd.concat([target0,target1])
# print(target['TenYearCHD'].value_counts())

data=target

X=data.iloc[:,0:15]
y=data.iloc[:,-1]

best=SelectKBest(score_func=chi2, k=10)
fit=best.fit(X,y)
data_scores=pd.DataFrame(fit.scores_)
data_columns=pd.DataFrame(X.columns)

scores=pd.concat([data_columns,data_scores],axis=1)
scores.columns=['Feature','Score']
print(scores.nlargest(11,'Score'))
# values har baar badal raha hai
scores=scores.sort_values(by="Score", ascending=False)
# feature selection
features=scores["Feature"].tolist()[:11]
print(features)
data=data[['sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp', 'diabetes', 'male', 'BPMeds', 'BMI','TenYearCHD']]
data=data.drop(['cigsPerDay'],axis='columns')
y = data['TenYearCHD']

X = data.drop(['TenYearCHD'], axis=1)
print(X.head())
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=1)
# We divide the dataset into training and test sub-datasets for predictive modeling
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

rf = RandomForestClassifier(n_estimators=200, random_state=0,max_depth=12)
rf.fit(train_x,train_y)

filen="randomf.pkl"
pickle.dump(rf,open(filen,'wb'))