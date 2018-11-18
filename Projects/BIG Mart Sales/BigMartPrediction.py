# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:33:49 2018

@author: Hrushikesh_M
"""

#BIG Mart
import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np
#matplotlib inline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import skew



train=pd.read_csv(r'data\train.csv');
test=pd.read_csv(r'data\test.csv');

#combine test and train data to do data preprocessing
all_data=pd.concat((train.loc[:,'Item_Identifier':'Outlet_Type'],
                      test.loc[:,'Item_Identifier':'Outlet_Type']));
                    

#Removing Unwanted columns
all_data.drop('Item_Identifier',axis=1,inplace=True);
#all_data.drop('Outlet_Establishment_Year',axis=1,inplace=True);
#all_data.drop('Outlet_Type',axis=1,inplace=True);




train.boxplot('Item_Outlet_Sales','Outlet_Identifier',figsize=(10,5));

#Lets use all other colmins

#Heat map to see what parameters affect sles most, seems price is an imporant factor
corr=train.corr()
sns.heatmap(corr,vmax=0.8)


#check for 
print(all_data.isnull().sum());

#for Outlet_Size , some Oulets dont have Outlet_Size so we cant get mesn. so lets put 
#it as 'NA'

all_data.Outlet_Size.fillna('NA',inplace=True);

#For Item_Weight, we replace nan by mean of its Item_type
all_data['Item_Weight'] = all_data['Item_Weight'].fillna(all_data.groupby('Item_Type')['Item_Weight'].transform('mean'));

all_data=pd.get_dummies(all_data);

s=train.shape
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

y = train.Item_Outlet_Sales

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
    
# Lets do simple Linear Regression first

linearRegressor = LinearRegression()
linearRegressor.fit(X_train,y);

yPrediction1 = linearRegressor.predict(X_test);

error1=rmse_cv(LinearRegression()).mean();
print(error1);
#Lets do Randomn FOrest

RF=RandomForestRegressor();
RF.fit(X_train,y);
yPrediction2 = RF.predict(X_test);
error2=rmse_cv(RandomForestRegressor()).mean();

print(error2);