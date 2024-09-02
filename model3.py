import loadTrainingdata as ltd
import functions as fns
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import datasets, linear_model

nvda = pd.read_csv('nvdadata.csv')
nvda = ltd.createTrainingDf(nvda, 8)
xtrain,ytrain,xtest,ytest = ltd.trainingTest(nvda, 3950, 40)
ytrain,ytest = ltd.linearToBinary(ytrain), ltd.linearToBinary(ytest)

def model31():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim = 8))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return model

m = model31()
print(fns.evalog(5,model=m,testdata=xtest,targets=ytest,xtr=xtrain,ytr=ytrain,valsplit=0.15,epocs=100))
