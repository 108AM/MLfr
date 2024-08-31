import loadTrainingdata as ltd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

nvda = pd.read_csv('nvdadata.csv')
nvdadf = ltd.createTrainingDf(nvda, 16)
xtrain,ytrain,xtest,ytest = ltd.trainingTest(nvdadf, 500, 2)

def model1():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape = (16,)))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mae'])
    model.fit(xtrain, ytrain, validation_split=0.15, epochs=10)

print(xtrain[0])


