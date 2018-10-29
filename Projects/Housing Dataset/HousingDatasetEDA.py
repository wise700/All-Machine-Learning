# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:16:03 2018

@author: Hrushi
"""

import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np
#matplotlib inline
from sklearn.decomposition import PCA


train=pd.read_csv(r'data\train.csv')
train.head()

train['SalePrice'].describe()
sns.set_style("whitegrid")
# Box plotting a varibale (SalePrice), gives us how the saleprice is distributed.

# sns.boxplot(y=train['SalePrice'])

# histogram
sns.distplot(train['SalePrice'])


print("Skweness is : %f" % train['SalePrice'].skew())

#always have column names handy
Cols=(train.columns.values)

#SalePrice with GrtLivArea

data=pd.concat([train['GrLivArea'],train['SalePrice']],axis=1);

data.plot.scatter(x='GrLivArea',y='SalePrice')


#SalePrice and TotalBsmtSF
data=pd.concat([train['TotalBsmtSF'],train['SalePrice']],axis=1);
#data.plot.scatter(x='TotalBsmtSF',y='SalePrice')



#Relationship with Categorical Vaaribales
data=pd.concat([train['OverallQual'],train['SalePrice']],axis=1)
sns.boxplot(x='OverallQual',y='SalePrice',data=data)

data=pd.concat([train['YearBuilt'],train['SalePrice']],axis=1)
data.plot.scatter(x='YearBuilt',y='SalePrice')

sns.boxplot(x='YearBuilt',y='SalePrice',data=data)

# Finding relationships with Density plots for categorical variables
#Lets look how the density plot looks for Categorocal variable Bulding type vs SalesPrice
BuildingTypes = train.Neighborhood.unique();

for building in BuildingTypes:
    subset=train[train['Neighborhood']==building]
    #Draw Density plot
    sns.distplot(subset['SalePrice'],hist=False,kde=True,label=building,kde_kws = {'linewidth': 3})
#Correlation Matrix

corr=train.corr()
sns.heatmap(corr,vmax=0.8)


#Zoomed heat map

cols1=corr.nlargest(10,'SalePrice')['SalePrice'].index
cm=np.corrcoef(train[cols1].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

cols=cols1.drop('SalePrice')

# Scatter plots , Jagger style
sns.pairplot(train[cols])


#Missing Data
#Finding out the cols with missing data with %
total=train.isnull().sum().sort_values(ascending=False);
percentage=total/1460*100

missing_Values=pd.concat([total,percentage],axis=1,keys=['total','percentage'])

#deleting all the cols where it has > 15% of missing values
colsToDelete=missing_Values[missing_Values['percentage']>15].index;
train=train.drop(colsToDelete,1)

train=train.drop(train.loc[train['Electrical'].isnull()].index);


#lets train a model
