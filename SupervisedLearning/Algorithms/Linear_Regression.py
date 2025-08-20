import pandas as pd
import numpy as np
import os
from  sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# df = pd.read_csv('cars24-car-price.csv')

class LinearRegression():
  def __init__(self,learning_rate=0.01,interations=5):
    self.learning_rate = learning_rate
    self.interations = interations

  def predict(self,X):
    return np.dot(X,self.W)+self.b
  
  def r2(self,X,y):
    y_ = self.predict(self,X)
    rss = np.sum((y-y_)**2)
    tss = np.sum((y-y.mean())**2)
    r2 = (1- rss/tss)
    return r2
  
  def update_weights(self):
    Y_pred = self.predict(self.X )
    # calculate gradients
    dW = - (2*(self.X.T ).dot(self.Y - Y_pred))/self.m
    db = - 2*np.sum(self.Y - Y_pred)/self.m
    # update weights
    self.W = self.W - self.learning_rate * dW
    self.b = self.b - self.learning_rate * db
    return self
  
  def fit(self, X, Y):
  # no_of_training_examples, no_of_features
    self.m, self.d = X.shape
    # weight initialization
    self.W = np.zeros(self.d)
    self.b = 0
    self.X = X
    self.Y = Y
    self.error_list=[]
    self.w_list=[]
    self.b_list=[]
    # gradient descent learning
    for i in range(self.interations):
        self.update_weights()
        self.w_list.append(self.W)
        self.b_list.append(self.b)
        Y_pred=X.dot(self.W)+self.b
        error=np.square(np.subtract(Y,Y_pred)).mean()
        self.error_list.append(error)
    return self
  
df = pd.read_csv("cars24-car-price.csv")
df_cleaned = df.select_dtypes (include = 'number')
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns = df_cleaned.columns)


X = df_scaled.drop('selling_price', axis=1)
y = df_scaled["selling_price"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr = LinearRegression(interations=100)
lr.fit(X_train,y_train)



