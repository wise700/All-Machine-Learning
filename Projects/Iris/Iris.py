# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:12:41 2018

@author: Hrushikesh_M
"""
# Import data and modules
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns;sns.set()
from sklearn.linear_model import LinearRegression
# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

df = sns.load_dataset('iris');

#iris = datasets.load_iris();

#x=iris.data;
#y=iris.target;
#
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0);
#
#
#
## use the function regplot to make a scatterplot, Sepal width vs Folwer
#ax=sns.regplot(x=X_train[:,0], y=y_train,);
#ax.set(xlabel='Sepal Width',ylabel='flower');
##sns.plt.show()
#
### use the function regplot to make a scatterplot, Sepal width vs Folwer
##ax=sns.regplot(x=X_train[:,1], y=y_train,);
##ax.set(xlabel='Sepal Length',ylabel='flower');
#
#
#linearRegressor=LinearRegression();
#
#linearRegressor.fit(X=X_train,y=y_train);
#
#Y_Pred=linearRegressor.predict(X_test);

#lets do EDA

# for classification, lets see if the data is blanced class

df['species'].value_counts();

#Pandas scatter plot for 2 features
df.plot(kind='scatter',x='sepal_length',y='sepal_width',);

#sns boxplot all feature
sns.boxplot(data=df,x='species',y='sepal_length');

#Voilin plot
sns.violinplot(data=df,x='species',y='sepal_length');

#Pandas boxplot
df.boxplot(by='species');


#Heatmap

sns.heatmap(df.corr(),annot=True);
x=df[['sepal_length','sepal_width','petal_length','petal_length']];
y=df['species'];
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25);


#svc
model = svm.SVC() #select the algorithm
model.fit(X_train,y_train);

y_pred=model.predict(X_test);


print('The accuracy of the SVM is:',metrics.accuracy_score(y_pred,y_test));

#Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train);

y_pred=model.predict(X_test);
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(y_pred,y_test));


comapare=[[y_pred,y_test]];




    






