# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:49:26 2018

@author: Hrushikesh_M
"""
import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression  

#Loan Prediction

#Get the data

train=pd.read_csv(r'data\train.csv');
test=pd.read_csv(r'data\test.csv');


#EDA

print(train['Loan_Status'].value_counts());

sns.heatmap(train.corr(),cmap='cubehelix_r',annot=True);

# Heatmap work only for numerical features


#removing unnnesesary columns (Loan id onde)
print(train.columns.values.tolist());

train.drop('Loan_ID',axis=1,inplace=True);
test.drop('Loan_ID',axis=1,inplace=True);

Y_Train=train['Loan_Status'];
train.drop('Loan_Status',axis=1,inplace=True);
#X_Train=pd.get_dummies(train);


#Handling nan values in Dataset
print(train.isnull().sum());

#Gender replace nan by male
print(train['Married'].value_counts());
train['Gender'].fillna('Male',inplace=True);
print(train['Gender'].isnull().sum());

#Married replaced nan by Yes

train['Married'].fillna('Yes',inplace=True);

#Dependents replace nan by 0
train['Dependents'].fillna('0',inplace=True);
print(train['Dependents'].value_counts());
      

print(train.isnull().sum());

#Self_Employed replace nan by No
print(train['Self_Employed'].value_counts());
train['Self_Employed'].fillna('No',inplace=True);

#Loan Amount replace nan by Mean of Loan aminuts
train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True);

#Loan Term replace nan by Mean of Loan Term
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace=True);
print(train['Credit_History'].value_counts());
      
#Credit_History replace nan by 1
train['Credit_History'].fillna(1,inplace=True);

#
##Model
#LG=LogisticRegression();
#
#LG.fit(X_Train,Y_Train);

