# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:49:26 2018

@author: Hrushikesh_M
"""
import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np


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


