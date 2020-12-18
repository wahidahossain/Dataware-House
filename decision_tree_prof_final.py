# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:06:52 2020

@author: Nafi-Wafi
"""


#step #2
print ("Step #2")
import pandas as pd
import os
path = "C:/Users/Nafi-Wafi/Downloads/"
filename = 'dataset2.xlsx'
fullpath = os.path.join(path,filename)
data_group6_i = pd.read_excel(fullpath,sep=',')

##----------------------------------------
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
##----------------------------------------



# a.	Display the column names
# b.	Display the shape of the data frame i.e number of rows and number of columns
# c.	Display the main statistics of the data
# d.	Display the types of columns
# e.	Display the first five records
# f.	Find the unique values of the class

print("values",data_group6_i.columns.values)
print("shape",data_group6_i.shape)
print("describe",data_group6_i.describe())
print("dtypes",data_group6_i.dtypes) 
print("head",data_group6_i.head(2))
print("unique",data_group6_i['Recommendation'].unique())

# step #3
print ("step #3")
colnames=data_group6_i.columns.values.tolist()
predictors=colnames[:2]
target=colnames[2]
import numpy as np
#create an (new column) array of 0,1, of length len, set value to true if less than else false .75 
data_group6_i['is_train'] = np.random.uniform(0, 1, len(data_group6_i)) <= .75
print(data_group6_i.head(2))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_group6_i[data_group6_i['is_train']==True], data_group6_i[data_group6_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


###############################################################


from sklearn.tree import DecisionTreeClassifier
dt_wahida = DecisionTreeClassifier(criterion='entropy',min_samples_split=90, random_state=11)
dt_wahida.fit(train[predictors], train[target])

####################################################

preds=dt_wahida.predict(test[predictors])
pd.crosstab(test['Rating'],preds,rownames=['Actual'],colnames=['Predictions'])

####################################################

from sklearn.tree import export_graphviz
with open('C:/Users/Nafi-Wafi/Downloads/dtree33.dot', 'w') as dotfile:
    export_graphviz(dt_wahida, out_file = dotfile, feature_names = predictors)
dotfile.close()

####################################################

X=data_group6_i[predictors]
Y=data_group6_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

####################################################

dt1_wahida = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
dt1_wahida.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data 
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_wahida, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
score

####################################################

### Test the model using the testing data
testY_predict = dt1_wahida.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

######################################################

import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']);
plt.show()

##############################Execise 1#######################

dt1_wahida = DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_split=20, random_state=99)
dt1_wahida.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data 
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_wahida, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_wahida.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']);
plt.show()




