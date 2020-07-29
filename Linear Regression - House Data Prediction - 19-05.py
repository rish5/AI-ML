#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# inbuild dataset 
from sklearn.datasets import load_boston


# In[6]:


data = load_boston()


# In[9]:


data.keys()


# In[11]:


boston_df = pd.DataFrame(data.data, columns=data.feature_names)


# In[12]:


boston_df


# In[13]:


boston_df ['MEDV'] = data.target


# In[14]:


boston_df.head()


# In[15]:


boston_df.shape


# In[16]:


boston_df.columns


# In[17]:


boston_df.isnull().sum()


# In[21]:


## Check if Y in normalized or Not ? - Assumption of LR Model
plt.figure(figsize=(12,8))
sns.distplot(boston_df['MEDV'], color = 'magenta')
plt.show()


# In[25]:


plt.figure(figsize=(15,10))
sns.heatmap(boston_df.corr(), annot= True)


# In[26]:


# MEDV vs all - Not select values closer to zero, exactly 1 or -1


# In[28]:


plt.figure(figsize=(12,8))
plt.scatter(boston_df['LSTAT'], boston_df['MEDV'])
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()


# In[29]:


boston_df.drop(['CHAS'], axis=1, inplace=True) 


# In[30]:


boston_df


# In[31]:


X = boston_df.drop(['MEDV'], axis=1)
Y = boston_df['MEDV']


# In[32]:


X


# In[33]:


Y


# ### Split the Data into Train & Test

# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3) 


# ### Linear Regression 

# In[108]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression() 


# In[109]:


# training 
lr_model.fit(X_train, Y_train)


# In[110]:


## score
lr_model.score(X_test, Y_test)


# ### Metrics of Linear Regression 

# In[111]:


predictions = lr_model.predict(X_test)


# In[112]:


## mean squared error 
from sklearn.metrics import r2_score, mean_squared_error
mean_squared_error(Y_test, predictions)


# In[113]:


## The closer it is to 1 , the better it is 
r2_score(Y_test, predictions)


# ### New Data Predictions 

# In[115]:


new_data = [[0.00632,18.0,2.31,0.538,6.575,88.2,4.0900,1.0,286.0,15.3,396.90,8.98]]


# In[116]:


lr_model.predict(new_data)


# In[1]:


pwd()


# In[ ]:




