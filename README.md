# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import needed libriaries like numpy , pandas and StandardScaler from sklern preprocessing and deifne linear regression like X1,y learning_rate=0.01 and num_iters=1000
2. Apply the linear regression function to the standardlized features X1_scaled and Y1_scaled  to obtain parameters.
3. prepare the new data and make predication using the trained models.
4. print the predited value found out from the regression analysis.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Narendran K
RegisterNumber:  212223230135
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)), X1]
  theta = np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)

    errors=(predictions -y ).reshape(-1,1)

    theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)

  return theta



data=pd.read_csv('50_Startups.csv',header=None)
print(data.head)

X=(data.iloc[1: , :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1: ,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

![ML ex03 output](https://github.com/user-attachments/assets/2dd74290-5a06-4888-a3be-3180be2b994c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
