# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:16:03 2018

@author: Hrushi
"""

import pandas as pd
import matplotlib
import seaborn as sns
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
data.plot.scatter(x='TotalBsmtSF',y='SalePrice')