import loadTrainingdata as ltd
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def evalog(iters, model, testdata, targets, xtr, ytr, valsplit, epocs):
    totalLoss = 0
    def loss(preds, targs):
        msd = 0
        for i in range(len(preds)):
            #print(predictions[i],targets[i])
            msd += (preds[i] - targs[i])**2
        return msd
    for i in range(iters):
        model.fit(xtr, ytr, validation_split=valsplit, epochs=epocs)
        predictions = model.predict(testdata)
        totalLoss += loss(predictions, targets)
    return (totalLoss/iters)

if __name__ == 'main':
    nvda = pd.read_csv('nvdadata.csv')
    nvdadf = ltd.createTrainingDf(nvda, 8)
    xtrain,ytrain,xtest,ytest = ltd.trainingTest(nvdadf, 2, 2)
    print('hi')
