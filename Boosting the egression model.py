#Impoting Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
#importing the dataset
bostonData = pd.read_csv('housing.data.txt',delim_whitespace = True,header = None)
colmns_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
bostonData.columns = colmns_names
#Explorint tha data 
bostonData.describe()
''' make a correlation'''
pd.options.display.float_format = '{:.3f}'.format
make_correlation = bostonData.corr()
plt.figure(figsize=(14,11))
sn.heatmap(make_correlation,annot = True)
plt.show()
#Fitting a regression model between two varriables(LSTAT,MEDV)
X = bostonData['LSTAT'].values.reshape(-1,1)
Y = bostonData['MEDV'].values
''' spliting the data'''
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(X,Y,test_size = 0.25 , random_state = 0)
'''Modeling'''
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)
y_predit = linear_regressor.predict(x_test)
'''Checking The Accuracy'''
linear_regressor.score(x_train,y_train)
#visualizing the regressor line
plt.Figure(figsize = (9,8))
sn.regplot(X,Y,color = 'red',robust = True)
plt.xlabel('% lower status of the population')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.show()
''' with adding the normal distribution'''
sn.jointplot(bostonData['LSTAT'].values,bostonData['MEDV'].values,bostonData,kind = 'reg', size = 7)
#classifying the lutiers and inliers 
'''fitting the ransam regressor'''
from sklearn.linear_model import RANSACRegressor
random_sample_consensus = RANSACRegressor()
random_sample_consensus.fit(X,Y)
'''classifing into in and out liers'''
inlier_area = random_sample_consensus.inlier_mask_
outlier_area =np.logical_not(inlier_area)
'''Setting the line Points'''

line_X = np.arange(0, 40, 1)
line_y_ransac = random_sample_consensus.predict(line_X.reshape(-1, 1))
#visualizing the classification 
sn.set(context = 'notebook',style = 'darkgrid')
plt.figure(figsize = (12,10))
plt.scatter(X[inlier_area], Y[inlier_area], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_area], Y[outlier_area],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000s")
plt.legend(loc='upper left')
plt.show()
