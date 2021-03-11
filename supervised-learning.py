#!/usr/bin/env python
# coding: utf-8

# ## Graduate Rotational Internship Programme
# ## Author : G V Sreekar
# ## Prediction Using Supervised learning

# In[5]:



#importing libraries that are required
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt


# In[6]:


#importing data
url='http://bit.ly/w-data'
data=pd.read_csv(url)
data.head()


# In[7]:



# data.shape

data.info()


# In[8]:


#statistics of the data
data.describe()


# In[9]:


#plotting data
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('scores')
plt.show()


# In[10]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[11]:


#importind ML models
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(xtrain,ytrain)


# In[12]:


Y= regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,Y)
plt.show()


# In[13]:


print(xtest) #Testing data-In Hours


# In[14]:



y_pred=regressor.predict(xtest) # predicting the scores
print(y_pred)


# In[15]:


df=pd.DataFrame({'Actual':ytest,'predicted':y_pred})
df


# In[16]:


#Let's predict for our value
Hours=9.25
Pred_value=regressor.predict([[Hours]])
print(Pred_value)


# In[17]:


from sklearn import metrics
ypred=regressor.predict(xtest)
print('Mean Absolute Error:',metrics.mean_absolute_error(ytest,ypred))


# In[18]:


print('Mean Squared Error:',metrics.mean_squared_error(ytest,ypred))


# In[ ]:




