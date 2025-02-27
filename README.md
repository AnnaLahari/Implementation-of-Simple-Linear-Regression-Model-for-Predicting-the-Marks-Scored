# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.Lahari
RegisterNumber:  212223230111
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,Y_pred,color="blue")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:

 # head

![image](https://github.com/user-attachments/assets/74234293-9b89-4cbc-b95b-d2020a18325b)


# tail

![image](https://github.com/user-attachments/assets/a6e9ca2d-f757-460e-9cfe-5c6da95e1e42)


# Segregating data to variables

![image](https://github.com/user-attachments/assets/a4296513-db53-4143-a3a1-fcc3a79281d1)

![image](https://github.com/user-attachments/assets/4c690609-8497-4d05-a66a-99431ff2d9ac)


# Displaying predicted values

![image](https://github.com/user-attachments/assets/0d23324e-3aea-4277-a1fc-c4b75c8f7743)


# Displaying actual values

![image](https://github.com/user-attachments/assets/20254e94-a282-4cd5-92bf-259fca3dea5a)


# MSE MAE RMSE

![image](https://github.com/user-attachments/assets/6f039261-42e3-4fb4-884a-7fa050fbefb1)


# Graph plot for training data

![image](https://github.com/user-attachments/assets/bd511156-814a-42ec-84d6-161be2fdbdeb)


# Graph plot for test data

![image](https://github.com/user-attachments/assets/776a6af2-aa09-4f71-9b57-e866b00381c3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
