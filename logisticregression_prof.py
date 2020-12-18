# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:36:49 2020

@author: Nafi-Wafi
"""


import pandas as pd
import os
#path = "E:\\Users\\Administrator\\Week_9"
path="C:/Users/Nafi-Wafi/Downloads/"
filename = 'dataset.xlsx'
fullpath = os.path.join(path,filename)
data_group6_b = pd.read_excel(fullpath,sep=';')
print(data_group6_b.columns.values)
print(data_group6_b.shape)
print(data_group6_b.describe())
print(data_group6_b.dtypes) 
print(data_group6_b.head(5))

#step 3
print("step #3")
print(data_group6_b['Dress_ID'].unique())
import numpy as np

import csv
import random
def save_data():
    randomList = []

    data_group6_c = data_group6_b.copy()
    randomListS = random.sample(range(1, len(data_group6_b)), 10)
    randomListL = random.sample(range(1, len(data_group6_b)), 100)
    randomListS.append(1)
    print("Printing list of 10 random numbers")
    print(randomList)
    for i in randomListS:
          print(i)
#          data_group6_c.iloc[i] = np.nan
          data_group6_c.iat[i,4] = np.nan
          
    for i in randomListL:
          print(i)
#          data_group6_c.iloc[i] = np.nan
          data_group6_c.iat[i,5] = np.nan
   
          
    save_path = os.path.join(path,"Attribute+DataSet.xlsx")  
    data_group6_c.to_csv(save_path, index = False,sep=';',quotechar='"', quoting=csv.QUOTE_ALL)
    print("missing values\n",data_group6_c.isnull().sum())
   
    
    print("Remove missing values")
    data_group6_c.drop(columns=['Size'],inplace=True)
#2- fill the missing values 
    print("Replace missing values\n")
    data_group6_c['Rating'].fillna((data_group6_c['Rating'].mean()), inplace=True)
    print("Check for missing values\n")
    print("save_data----eeeee------------")
    print(len(data_group6_c)-data_group6_c.count())
    print("save_data-----------------------------------")
######################################################### 
#########################################################   
    data_group6_c['Recommendation']=(data_group6_c['Recommendation']=='1').astype(int)
#find the columns that have categories
    print("print categories\n")
    categoricals = []
    for col, col_type in data_group6_c.dtypes.iteritems():
       if col_type == 'O':
          categoricals.append(col)
       else:
          data_group6_c[col].fillna(0, inplace=True)
    print(categoricals)

 #############################################################   
    import matplotlib.pyplot as plt
    pd.crosstab(data_group6_c.Dress_ID,data_group6_c.Recommendation)
    pd.crosstab(data_group6_c.Dress_ID,data_group6_c.Recommendation).plot(kind='bar')
    plt.title('Purchase Frequency for Education Level')
    plt.xlabel('Dress_ID')
    plt.ylabel('Frequency of Purchase')
    print("save_data----------11111111111111-----------")
    print("Scatter plot with kernel density function\n")
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.pairplot(data_group6_c, diag_kind = 'kde')
    plt.show()
    print("line plot")
    sns.lineplot(data=data_group6_c,x='Dress_ID',y='Recommendation')
    plt.show()
    
    import matplotlib.pyplot as plt
    plt.figure(10)
    plt.hist(data_group6_c['Style'])
    plt.title('Histogram of Style')
    plt.xlabel('Size')
    plt.ylabel('Frequency')

#
# collapse 
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'Sexy', data_group6_b['Style'])
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'cute', data_group6_b['Style'])
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'Brief',data_group6_b['Style'])
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'Novelty', data_group6_b['Style'])
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'bohemian', data_group6_b['Style'])
data_group6_b['Dress_ID']=np.where(data_group6_b['Style'] =='Casual', 'party', data_group6_b['Style'])

# Modify file and save it.

save_data()
#os.system.exit()
# save_path = fullpath = os.path.join(path,"bank2.csv")
# data_group6_b.to_csv(save_path, index = False)
#Check the values of who  purchased the deposit account
print(data_group6_b['Recommendation'].value_counts())
#Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(data_group6_b.groupby('Recommendation').mean())
#Check the mean of all numeric columns grouped by Style
print("Group by Style -------------------")
print(data_group6_b.groupby('Style').mean())

#Plot a histogram showing purchase by education category
import matplotlib.pyplot as plt
pd.crosstab(data_group6_b.Dress_ID,data_group6_b.Recommendation)
pd.crosstab(data_group6_b.Dress_ID,data_group6_b.Recommendation).plot(kind='bar')
plt.title('Purchase Frequency for Dress Level')
plt.xlabel('Dress_ID')
plt.ylabel('Recommendation')
#draw a stacked bar chart of the Style and the Recommendation
table=pd.crosstab(data_group6_b.Style,data_group6_b.Recommendation)
#-----------------------------------------
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Style vs Recommendation')
plt.xlabel('Style')
plt.ylabel('Proportion of Customers')
#plot the bar chart for the Frequency of Purchase against Material
pd.crosstab(data_group6_b.Style,data_group6_b.Recommendation).plot(kind='bar')
plt.title('Purchase Frequency based on Style')
plt.xlabel('Style')
plt.ylabel('Recommendation')
#Repeat for the month
pd.crosstab(data_group6_b.Style,data_group6_b.Recommendation).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Style')
plt.ylabel('Frequency of Purchase')
#Plot a histogram of the age distribution
data_group6_b.Season.hist()
plt.title('Histogram of Season')
plt.xlabel('Season')
plt.ylabel('Frequency')
#Step #4
print("step #4a")
#Deal with the categorical variables, use a for loop
#1- Create the dummy variables 
print("before dummy")
print(data_group6_b.columns.values)
cat_vars=['Dress_ID','Style','NeckLine','Price','Season','Size',
          'SleeveLength','waiseline','Material','FabricType','Decoration','PatternType']
for var in cat_vars:
#    cat_list='var'+'_'+var
    cat_list=[]
    cat_list = pd.get_dummies(data_group6_b[var], prefix=var)
    print("after dummies\n",cat_list)    
    data_group6_b1=data_group6_b.join(cat_list)
    data_group6_b=data_group6_b1
print("after dummy supplied")
print(data_group6_b.columns.values)   
#  2- Removee the original columns
print ("remove the original columns")
############### imp ######################
cat_vars=['Dress_ID','Style','NeckLine','Price','Season','Size',
          'SleeveLength','waiseline','Material','FabricType','Decoration','PatternType']
data_peterb_b_vars=data_group6_b.columns.values.tolist()
print("step2: removecolumns")
to_keep=[i for i in data_peterb_b_vars if i not in cat_vars]
data_peterb_b_final=data_group6_b[to_keep]
data_peterb_b_final.columns.values
# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
print("step #4b")
data_peterb_b_final_vars=data_peterb_b_final.columns.values.tolist()
Y=['Recommendation']
X=[i for i in data_peterb_b_final_vars if i not in Y ]
print("columns Y",Y)
print("columns X",X)


############################################################
########################OOOOKKKKKKK######################
############################################################


#Step 5
print("step #5")

#1- We have many features so let us carryout feature selection
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs',max_iter=2000)
rfe = RFE(model, 5)
rfe = rfe.fit(data_peterb_b_final[X],data_peterb_b_final[Y].values.ravel() )

print(rfe.support_)
print(rfe.ranking_)
#2- Update X and Y with selected features which are manually identified from the rfe.
cols=['Style_party', 'FabricType_poplin', 'FabricType_batik', 'NeckLine_boat-neck' ] 
X=data_peterb_b_final[cols]
Y=data_peterb_b_final['Recommendation']
type(Y)
type(X)



#step 6
print("step 6")
#1- split the data into 70%training and 30% for testing, note  added the solver to avoid warnings
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 2-Let us build the model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
clf1.fit(X_train, Y_train.values.ravel())
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print("probs")
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print("Accuracy of the model:-------------------------")
print (metrics.accuracy_score(Y_test, predicted))	






#step 7
print("step 7")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs',max_iter=2000), X, Y, scoring='accuracy', cv=10)
print (scores)
print("Mean:")
print (scores.mean())
#step 8
print("step #8")
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
import numpy as np
#Y_A =Y_test.values
Y_A = pd.Series(np.where(Y_test.values == 'yes',1,0),Y_test.index)
#sample.housing.eq('yes').mul(1)
##print('Y_A',Y_A)
Y_P = np.array(prob_df['predict'])
##print('Y_P',Y_P)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_A, Y_P)
##print (confusion_matrix)

##############    confusion_matrix        ############
y=data_peterb_b_final['Recommendation']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix ---------------------------------------")
print(confusion_matrix)

######################################





