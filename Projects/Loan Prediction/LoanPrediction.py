# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:49:26 2018

@author: Hrushikesh_M
"""
import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv

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

all_data=train.append(test,ignore_index=True);

############################Missing Values############################
#Handling nan values in Dataset
print(all_data.isnull().sum());

#Gender replace nan by male
print(all_data['Married'].value_counts());
all_data['Gender'].fillna('Male',inplace=True);
print(all_data['Gender'].isnull().sum());

#Married replaced nan by Yes

all_data['Married'].fillna('Yes',inplace=True);

#Dependents replace nan by 0
all_data['Dependents'].fillna('0',inplace=True);
print(all_data['Dependents'].value_counts());
      
#Self_Employed replace nan by No
print(all_data['Self_Employed'].value_counts());
all_data['Self_Employed'].fillna('No',inplace=True);

#Loan Amount replace nan by Mean of Loan aminuts
all_data['LoanAmount'].fillna(all_data['LoanAmount'].mean(),inplace=True);

#Loan Term replace nan by Mean of Loan Term
all_data['Loan_Amount_Term'].fillna(all_data['Loan_Amount_Term'].mean(),inplace=True);
print(all_data['Credit_History'].value_counts());
      
#Credit_History replace nan by 1
all_data['Credit_History'].fillna(1,inplace=True);

print(all_data.isnull().sum());

############################Missing Values############################


############################Categorical Variables######################

all_data_with_dummies=pd.get_dummies(all_data);

##with Label encoder you can do only one column at a time, then use ohe to get same as pd.e=get_dummies
LE =LabelEncoder();
#LE_data_ohe = LE.fit_transform(all_data['Married']);

#encode the output also
Y_Train=LE.fit_transform(Y_Train);
############################Categorical Variables######################

#Split train test

X_Train=all_data_with_dummies[:train.shape[0]];
Y_Test=all_data_with_dummies[train.shape[0]:];


##Model
LG=LogisticRegression();

LG.fit(X_Train,Y_Train);

pred=LG.predict(Y_Test);

prede=list(LE.inverse_transform(pred));

with open('pred.csv', 'w',newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(prede)
    
writeFile.close()

