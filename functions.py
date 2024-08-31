import loadTrainingdata as ltd
import pandas as pd

def loss(predictions, targets):
    msd = 0
    for i in range(len(predictions)):
        print(predictions[i],targets[i])
        msd += (predictions[i] - targets[i])**2
    return msd

nvda = pd.read_csv('nvdadata.csv')
nvdadf = ltd.createTrainingDf(nvda, 8)
xtrain,ytrain,xtest,ytest = ltd.trainingTest(nvdadf, 2, 2)

print(xtest[0], ytest[0], xtest[1], ytest[1])