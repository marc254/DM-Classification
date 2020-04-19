# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:40:50 2020

@author: skambou
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.contingency import chi2_contingency
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import warnings

## Part 1 Classification
### Data Description
#### Rows: 7299
#### Columns : 27


df = pd.read_csv('E:\\Marc\\Data Mining\\Project\\Classification\\Amazon and Best Buy Electronics summary.csv')
df.count()

df.isnull().sum()

##### Important columns for the analysis are: brand, reviews.dorecommend, reviews.rating

df.brand.unique()
df.brand.describe()
df.info()

#Encoding brand

cleanup_nums = {"brand": {'Microsoft':1, 'Boytone':2, 'Sanus':3, 'Ultimate Ears':4, 
                              'Lowepro':5, 'Corsair':6, 'Sdi Technologies, Inc.':7, 
                              'Verizon Wireless':8, 'JVC':9, 'JBL':10, 'Lenovo':11, 
                              'Siriusxm':12, 'Pny':13, 'Sling Media':14, 'Sony':15,
                              'Midland':16, 'Toshiba':17, 'Power Acoustik':18, 'House of Marley':19,
                              'Yamaha':20, 'DreamWave':21, 'Glengery':22, 'Dell':23, 'MEE audio':24, 
                              'Samsung':25, 'Bose':26, 'Logitech':27, 'Motorola':28, 
                              'Definitive Technology':29, 'Alpine':30, 'Belkin':31, 
                              'Bowers & Wilkins':32, 'CLARITY-TELECOM':33, 'Kicker':34, 'SVS':35,
                              'WD':36, 'Netgear':37, 'Peerless-AV':38}}

df2 = df

df2.replace(cleanup_nums, inplace=True)


def rec(x):
    if x == True: return 1
    else: return 0
    
df2['recommend'] = df2['reviews.doRecommend'].apply(rec)

df2['reviews.rating'].fillna(df2['reviews.rating'].median(), inplace=True)

df2.columns
#df['recommend'] = df['reviews.doRecommend']



## Steps for modeling

### 1) import libraries (pandas, numpy, seaborn, sklearn, 
### 2) import dataset
### 3) See a quick summary of the dataset (pandas describe) and corraelation heatmap (use seaborn)
### 4) Check for missing values and handle them
### 5) Convert categorical variables to numeric values using encoding
### 6) Do scaling if required
### 7) Extract features (independent) and responses variables separately
### 8) Do correlation and analyses
### 9) Choose appropriate model and import from sklearn
### 10) Divide the dataset into training and test set
### 11) Train the model on the training set
### 12) Evaluate model on the test set


X = df2.loc[:,['brand', 'reviews.rating']]
y = df2.loc[:,['recommend']]

#X= np.array(df1['reviews.rating']).reshape(-1,1)


#split X and y in test sample and train sample
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)


#preprocessing
#sc = StandardScaler()
#sc.fit(X_train)
#X_train_scaled = sc.transform(X_train)
#X_test_scaled = sc.transform(X_train)



import statsmodels.api as sm
X2_train = sm.add_constant(X_train)
X2_test = sm.add_constant(X_test)
ols = sm.Logit(y_train,X2_train)
lr = ols.fit()


while lr.pvalues.max()>0.05:
    X2_train=np.delete(X2_train,lr.pvalues.argmax(),axis=1)
    X2_test=np.delete(X2_test,lr.pvalues.argmax(),axis=1)
    ols = sm.Logit(y_train,X2_train)
    lr = ols.fit()

print(lr.summary())


#training the model
model = LogisticRegression()
model.fit(X_train, y_train)

model.intercept_
model.coef_


#Scale the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#test the model
#y_pred = model.predict(X_test)

#use model to predict
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#performance measures
cm = confusion_matrix(y_test, y_pred_test)
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
sensitivity = recall_score(y_test, y_pred_test)
print(classification_report(y_test, y_pred_test))

print('Accuracy:', accuracy)

print('Precision:', precision)

# Confusion Matrix

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
plt.title('Confusion Matrix/n')
plt.show()




#k-fold cross_validation 
#preprocessing
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

scores = cross_val_score(LogisticRegression(),X_scaled,y,cv=4,scoring='precision')
print('Cross-validated scores:',scores)
scores.mean()


#Plotting ROC

y_pred_probs = model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test,y_pred_probs)

plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')

plt.title('ROC with AUC score: {}'.format(roc_auc_score(y_test,y_pred_probs)))
plt.show()