#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas_ta as pta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_indicators(data, ema1=200, ema2=50, ema3=20, stoch=14, rsi=14, macd_f=12, macd_sl=26, macd_si=9, cci=14,
                   bb_tmpr=5, bb_up=2, bb_dwn=2, atr=14):
    dataframe = data.copy()
    
    #EMA (200,50,20)
    dataframe[f'{ema1}ema'] = dataframe['Adj Close'].ewm(span=ema1).mean()
    dataframe[f'{ema2}ema'] = dataframe['Adj Close'].ewm(span=ema2, adjust=False).mean()
    dataframe[f'{ema3}ema'] = dataframe['Adj Close'].ewm(span=ema3, adjust=False).mean()

    #STOCHASTIC
    dataframe['14-high'] = dataframe['High'].rolling(stoch).max()
    dataframe['14-low'] = dataframe['Low'].rolling(stoch).min()
    dataframe['%K'] = (dataframe['Adj Close'] - dataframe['14-low'])*100/(dataframe['14-high'] - dataframe['14-low'])
    dataframe['%D'] = dataframe['%K'].rolling(3).mean()

    #RSI (RELATIVE STRENGTH INDEX)
    dataframe['RSI'] = pta.rsi(dataframe['Adj Close'], length = rsi)

    #MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
    macd = pta.macd(dataframe['Adj Close'], fastperiod=macd_f, slowperiod=macd_sl, signalperiod=macd_si)
    dataframe['MACD'] = macd.iloc[:,0].values
    dataframe['MACDh'] = macd.iloc[:,1].values
    dataframe['MACDs'] = macd.iloc[:,2].values

    #CCI (COMMODITY CHANNEL INDEX)
    cci_val = pta.cci(dataframe['High'], dataframe['Low'], dataframe['Adj Close'], timeperiod=cci)
    dataframe['CCI'] = cci_val.values

    #PSAR (PARABOLIC SAR)
    psar = pta.psar(dataframe['High'], dataframe['Low'])
    a = pd.concat([psar.iloc[:,0],psar.iloc[:,1]*-1], axis=1)
    dataframe['PSAR'] = a.iloc[:,0].fillna(a.iloc[:,1])

    #BB (BOLLINGER BANDS)
    bbands = pta.bbands(dataframe['Adj Close'], timeperiod=bb_tmpr, nbdevup=bb_up, nbdevdn=bb_dwn, matype=0)
    dataframe['BBl'] = bbands.iloc[:,0].values
    dataframe['BBm'] = bbands.iloc[:,1].values
    dataframe['BBu'] = bbands.iloc[:,2].values
    dataframe['BBb'] = bbands.iloc[:,3].values
    dataframe['BBp'] = bbands.iloc[:,4].values

    #ATR (AVERAGE TRUE RANGE)
    atr = pta.atr(dataframe['High'], dataframe['Low'], dataframe['Adj Close'], timeperiod=atr)
    dataframe['ATR'] = atr.values
    
    return dataframe



def up_down(data):
    dataframe = data.copy()

    dataframe['log_ret'] = dataframe['Adj Close'].pct_change()

    lista =[]

    for index, row in dataframe.iterrows():
        if row['log_ret'] > 0:
            lista.append(1)
        else:
            lista.append(0)

    serie = pd.Series(lista, index = dataframe.index)

    dataframe['Up_Down'] = serie
    
    return dataframe

def full_prediction(yhat, test_X, test_y):
    yhat_re = yhat.reshape((yhat.shape[0], yhat.shape[1]))
    test_X_re = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.concatenate((yhat_re, test_X_re), axis=1)

    test_y_re = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y_re, test_X_re), axis=1)

    return inv_y, inv_yhat

def full_prediction_binary(yhat, test_X, test_y):
    yhat_re = yhat
    test_X_re = test_X
    inv_yhat = np.concatenate((yhat_re, test_X_re), axis=1)

    test_y_re = test_y
    inv_y = np.concatenate((test_y, test_X), axis=1)

    return inv_y, inv_yhat

def result(inv_yhat, inv_y):
    res = pd.DataFrame({'yhat':inv_yhat[:,0], 'y':inv_y[:,0]})

    res['yhat_log'] = res['yhat'].pct_change()
    res['y_log'] = res['y'].pct_change()

    res['res'] = res['yhat_log']*res['y_log']

    lista =[]
    for index, row in res.iterrows():
        if row['res'] > 0:
            lista.append(1)
        else:
            lista.append(0)

    serie = pd.Series(lista, index = res.index)

    res['Resultado'] = serie

    return res



# In[ ]:




