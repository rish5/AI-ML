#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


diabetes_df = pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/diabetes.csv')


# In[5]:


diabetes_df.head(10)


# ### DAta Manipulation

# In[4]:


diabetes_df.isnull().sum()


# In[6]:


X = diabetes_df.drop(['Pregnancies', 'Outcome'], axis=1)


# In[7]:


X.head()


# ### Replace zero with nan values

# In[8]:


X.replace(0,np.nan,inplace=True)


# In[9]:


X.isnull().sum()


# In[11]:


## use simple Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# In[12]:


diabetes_array = imputer.fit_transform(X)


# In[13]:


diabetes_array


# In[15]:


diabetes_df2 = pd.DataFrame(diabetes_array, columns=X.columns)


# In[16]:


diabetes_df2


# In[17]:


diabetes_df2.isnull().sum()


# In[20]:


diabetes_df2['Pregnancies'] = diabetes_df.Pregnancies


# In[21]:


diabetes_df


# In[22]:


Y = diabetes_df['Outcome']


# ### Cross Validation
# * Splitting data into Train and Test

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(diabetes_df2,Y,test_size=0.2,random_state=0)


# ### logistic regression

# In[34]:


from sklearn.linear_model import LogisticRegression
# C = just like SVM, higher the C better will be decision boundary
# max_iteration = higher the better
#solver = for binary class classification - use 'liblinear'
#penalty = regularization l1 or l2 - for liblinear use l1, for rest solver use l2
logit_reg = LogisticRegression(C=1e7, max_iter=1e9, solver='liblinear',penalty='l1')


# In[35]:


logit_reg.fit(X_train,Y_train)
#C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
#STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
#Increase the number of iterations (max_iter) or scale the data as shown in:
#    https://scikit-learn.org/stable/modules/preprocessing.html
#Please also refer to the documentation for alternative solver options:
#    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)


# In[40]:


logit_reg.score(X_test, Y_test)


# In[41]:


## Predictions
yhat = logit_reg.predict(X_test)


# In[42]:


yhat


# ### Classification Report

# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, yhat)


# In[45]:


sns.heatmap(confusion_matrix(Y_test, yhat), annot=True)
# correction_predict = 96+28


# In[46]:


(96+28)/(97+10+19+28)


# In[47]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, yhat))


# In[ ]:


'''
precison = ratio b/w correctly positive observation of the total predicted positive obervation
Precision = TP/TP+FP

recall = sensitivity 
recall = ratio b/w correctly predicted positive observation to the all observation in actual class = yes
recall = TP/FP+FN

F1 score 
F1 score is the weighted avg. of recision and recall
F1 score = 2*(recall*precision)/recall+precision

support =  total no. of obs.
'''

