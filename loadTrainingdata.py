import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def createCSV(tickerName, fileName):
    #creates csv of stock data
    stock = yf.Ticker(tickerName)
    stockHS = stock.history(period = "1mo", interval = "2m")
    stockHS.to_csv(fileName)

def plotClosingStocks(pddf, start=0, end=None):
    #plots time series of stocks
    if end is None:
        end = len(pddf)
    dates = [pddf['Datetime'].iloc[i] for i in range(start,end)]
    closes = [pddf['Close'].iloc[i] for i in range(start,end)]
    times = [time.split('-')[0] for time in [datetime.split(' ')[1] for datetime in dates]]
    plt.plot(times,closes)
    plt.show()

def copyCSV(fileName, newFileName):
    #copy csv file
    if fileName != newFileName:
        df0 = pd.read_csv(fileName)
        df0.to_csv(newFileName)

def createTrainingDf(df0, numberFeatures):
    #create dataframe of gain from last time period from pandas dataframe of stock data
    df0['gain'] = df0['Close'].shift(-1) - df0['Close']
    for i in range(numberFeatures):
        df0[str(i+1)] = df0['gain'].shift(i+1)
    return df0

def trainingTest(df, trainExamples, testExamples):
    #returns training and test data and target
    index = df.columns.get_loc('Stock Splits')
    features = len(df.columns)-index-2
    xtrain = df.loc[[i for i in range(features,trainExamples+features)], [str(i) for i in range(1,features+1)]].to_numpy()
    ytrain = df.loc[[i for i in range(features,trainExamples+features)], ['gain']].to_numpy()
    xtest = df.loc[[i for i in range(3000,testExamples+3000)], [str(i) for i in range(1,features+1)]].to_numpy()
    ytest = df.loc[[i for i in range(3000,trainExamples+3000)], ['gain']].to_numpy()
    return(xtrain,ytrain,xtest,ytest)


