import loadTrainingdata as ltd
import functions as fns
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import datasets, linear_model

nvda = pd.read_csv('nvdadata.csv')
nvdadf = ltd.createTrainingDf(nvda, 5)
xtrain,ytrain,xtest,ytest = ltd.trainingTest(nvdadf, 2950, 100)

def model21(testdata):
    regr = linear_model.LinearRegression()
    regr.fit(xtrain, ytrain)
    preds = regr.predict(testdata)
    return preds

def model20(testdata):
    model = Sequential()
    model.add(Dense(1, activation='relu', input_shape = (5,)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mae'])
    model.fit(xtrain, ytrain, validation_split=0.15, epochs=100)
    m = model.predict(testdata)
    return m

preds = model20(xtest)
print(fns.eval(100,preds,ytest))