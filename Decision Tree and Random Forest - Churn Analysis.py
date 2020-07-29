#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


churn_df = pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/churn-bigml-20.csv')


# In[3]:


churn_df.head()


# In[5]:


# True = Left the service & False = Continiued with the same
sns.countplot(x = 'Churn', data = churn_df)


# In[8]:


churn_df.info()


# In[6]:


churn_df['International plan'].unique()


# In[7]:


churn_df['Voice mail plan'].unique()


# In[10]:


## Replace Yes & No with 0 & 1
churn_df['International plan'].replace('Yes', 1, inplace=True)


# In[11]:


churn_df['International plan'].replace('No', 0, inplace=True)


# In[12]:


churn_df['International plan'].unique()


# In[13]:


## Replace Yes & No with 0 & 1
churn_df['Voice mail plan'].replace('Yes', 1, inplace=True)


# In[14]:


churn_df['Voice mail plan'].replace('No', 0, inplace=True)


# In[15]:


churn_df.head()


# In[17]:


churn_df['Voice mail plan'].replace('No', 0, inplace=True)


# ## Correlation

# In[18]:


plt.figure(figsize=(20,12))
sns.heatmap(churn_df.corr(), annot= True)


# ## Split Data Into Train and Test

# In[19]:


X = churn_df.drop(['Churn'], axis=1)


# In[20]:


Y = churn_df['Churn']


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# ## Decision Tree

# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


clf = DecisionTreeClassifier(criterion='entropy', max_depth=9)


# In[44]:


clf.fit(X_train, Y_train)


# In[45]:


clf.score(X_test, Y_test)


# In[46]:


clf.score(X_train, Y_train)


# In[47]:


predictions = clf.predict(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)


# In[49]:


sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, fmt='0.0f')


# ## Ensemble Learning

# In[30]:


## Random Forest


# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:




