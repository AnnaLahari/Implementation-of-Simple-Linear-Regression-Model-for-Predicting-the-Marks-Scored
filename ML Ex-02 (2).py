#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
print('NAME: A.LAHARI')
print('REG.No : 212223230111')


# In[2]:


df.tail()


# In[3]:


X=df.iloc[:,:-1].values
X


# In[4]:


Y=df.iloc[:,1].values
Y


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


# In[7]:


Y_pred


# In[8]:


Y_test
print('NAME: A.LAHARI')
print('REG.No : 212223230111')


# In[9]:


print('NAME: A.LAHARI')
print('REG.No : 212223230111')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


# In[10]:


plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[11]:


plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,Y_pred,color="blue")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

