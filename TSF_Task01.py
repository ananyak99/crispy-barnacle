#!/usr/bin/env python
# coding: utf-8

# ## The Sparks Foundation - GRIP -  Data Science & Business Analytics - Nov'2020 

# ### Task 01: Prediction using Supervised ML

# #### Author: Ananya Kamalapur

# ### Problem Statement(s):
# ####  1. Predict the percentage of a student based on the no. of study hours.
# #### 2. What will be the predicted score if a student studies for 9.25 hours/day?
# 

# ### Implementation:

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import seaborn as sns
import matplotlib.pyplot


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.ExcelFile(r'C:\Users\Ananya Kamalapur\Desktop\task01.xlsx')
data1=data.parse()
data1.head()


# In[5]:


df = pd.read_excel (r'C:\Users\Ananya Kamalapur\Desktop\task01.xlsx')
print (df)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


sns.distplot(df['Scores'])


# In[9]:


X=df['Hours']
y=df['Scores']


# In[10]:


X=np.array(X)


# In[11]:


X=X.reshape(-1,1)


# ### Splitting the dataset into training dataset and testing dataset

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=50)


# ### Linear Regression Model 

# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


l=LinearRegression()


# In[16]:


l.fit(X_train, y_train)


# In[17]:


print(l.intercept_)


# In[18]:


l.coef_


# In[19]:


p=l.predict(X_test)


# In[20]:


p


# In[21]:


plot_df=pd.DataFrame({'y_test':y_test, 'p':p})


# In[22]:


sns.regplot(x='y_test', y='p', data=plot_df, fit_reg=True)


# In[23]:


sns.distplot(y_test-p)


# In[24]:


from sklearn import metrics


# # Evaluating the Model

# In[26]:


print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test,p)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test,p)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test,p))))


# # Prediction of percentage if a student studies for 9.25 hours/day

# In[27]:


print('No. of study hours: {} hours'.format(9.25))
print('Predicted Score: {} %'.format(l.predict([[9.25]])))


# # THE END!
