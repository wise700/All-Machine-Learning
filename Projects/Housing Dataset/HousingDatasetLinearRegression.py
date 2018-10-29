import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np
#matplotlib inline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


train=pd.read_csv(r'data\train.csv')
test=pd.read_csv(r'data\test.csv')

all_data=pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']));
                    
all_data.shape

corr=train.corr()
sns.heatmap(corr,vmax=0.8)


#Zoomed heat map, to know the  most important variables

cols1=corr.nlargest(10,'SalePrice')['SalePrice'].index
cm=np.corrcoef(train[cols1].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols1.values, xticklabels=cols1.values)

cols=cols1.drop('SalePrice')


#log transform

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#Convert categorical variables

all_data = pd.get_dummies(all_data);

#Fill na with mean
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
s=train.shape
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

y = train.SalePrice


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
    
# Lets do simple Linear Regression first

linearRegressor = LinearRegression()
linearRegressor.fit(X_train,y);

yPrediction = linearRegressor.predict(X_test)

error=rmse_cv(LinearRegression()).mean()

#Ridge Regression
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
            
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")