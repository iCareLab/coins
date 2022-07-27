import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit

def load_pure_data(ticker, db_path, split=0.8, verbose=False, plot=False):
    """
    Getting Pure Data
    Input : .cvs data format, OHLCV
    Output: X data & Y data
    """
    if (ticker is None) or (db_path is None):
        print('Cannot Open db file')
        return None

    df = pd.read_csv(db_path + ticker + '.csv', index_col='Unnamed: 0', parse_dates=True)
    #print(df)

    # df : 'open', 'high', 'low', 'close', 'volume', 'value'
    df = df.drop(columns='value')   # 'open', 'high', 'low', 'close', 'volume'
    df = df.drop(columns='volume')   # 'open', 'high', 'low', 'close', 'volume'
    if verbose is True: print('Original data:\n', df)

    ''' Split Data Set to train data and test data '''
    SPLIT = int(len(df.index) * split)
    split_train = df.iloc[:SPLIT]
    split_test  = df.iloc[SPLIT:]
    if verbose is True: print('train set:', np.array(split_train.shape))
    if verbose is True: print('test set: ', np.array(split_test.shape))

    if plot is True:
        #plt.plot(df['close'])
        plt.plot(split_train['close'])
        plt.plot(split_test['close'])
        plt.show()

    return (df, split_train, split_test)

def make_data_set(data, window_size=None, verbose=False):
    ''' Scaled Normalize '''
    if verbose is True: print('Train Data Set:')
    X_scaled = StandardScaler().fit_transform(data)
    Y_scaled = MinMaxScaler().fit_transform(data[['close']])    # 'close' column
    if verbose is True: print('Scaled X:\n', X_scaled[:5], X_scaled.shape)
    if verbose is True: print('Scaled Y:\n', Y_scaled[:5], Y_scaled.shape)

    ''' Make split by windows size '''
    X_data = []
    for i in range(window_size, len(X_scaled)-1):
        X_data.append(X_scaled[i-window_size:i, :])

    Y_data = Y_scaled[window_size:-1]

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    if verbose is True: print('X-data set:\n', X_data[:5], X_data.shape)
    if verbose is True: print('Y-data set:\n', Y_data[:5], Y_data.shape)

    return [X_data, Y_data]

def np_array2Tensor(X=None, Y=None, verbose=False):
    tensor_X = Variable(torch.Tensor(X))
    tensor_Y = Variable(torch.Tensor(Y))
    if verbose is True: print("tensor data set:", tensor_X.shape, tensor_Y.shape)

    return tensor_X, tensor_Y



## 변수의 상관 관계 분석
import matplotlib.pyplot as plt
import statsmodels.api as sm    #Modeling algorithms -- General
from scipy import stats

def correlation_analysis(verbose=False, Plot=False):
    (_df, X, y) = load_pure_data(ticker, db_path, verbose = False)

    df = _df.copy()
    df = df.asfreq('H', method = 'ffill') #결측치가 있으면 앞의 데이터를 이용해서 채워 넣는다.
    result = sm.tsa.seasonal_decompose(df['close'], model='additive')
    Y_trend = pd.DataFrame(result.trend)
    Y_trend.fillna(method='ffill', inplace=True)
    Y_trend.fillna(method='bfill', inplace=True)
    Y_trend.columns = ['trend']

    Y_seasonal = pd.DataFrame(result.seasonal)
    Y_seasonal.fillna(method='ffill', inplace=True)
    Y_seasonal.fillna(method='bfill', inplace=True)
    Y_seasonal.columns = ['seasonal']

    Y_day = df[['close']].rolling(24).mean()
    Y_day.fillna(method='ffill', inplace=True)
    Y_day.fillna(method='bfill', inplace=True)
    Y_day.columns = ['day']

    Y_week = df[['close']].rolling(24*7).mean()
    Y_week.fillna(method='ffill', inplace=True)
    Y_week.fillna(method='bfill', inplace=True)
    Y_week.columns = ['week']

    df = pd.concat([df, Y_trend, Y_seasonal, Y_day, Y_week], axis=1)

    #df['Year'] = df.index.year
    #df['Quarter'] = df.index.quarter
    #df['Quarter_ver2'] = df['Quarter'] + (df.Year - df.Year.min()) * 4
    #df['Month'] = df.index.month
    #df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    #df['WorkDay'] = df.index.dayofweek
    #df = df.asfreq('H', method = 'bfill') #결측치가 있으면 앞의 데이터를 이용해서 채워 넣는다.
    if verbose is True: print(df.info())
    if verbose is True: print(df.describe(include='all').T)

    fit_df = sm.OLS(df['close'], df).fit()
    if verbose is True: print(fit_df.summary())

    if Plot is True:
        print(df.corr())
        #df.hist(bins=20, grid=True, figsize=(16, 12))
        df.boxplot(column='volume', by='Hour', grid=True, figsize=(12,5))
        df.boxplot(column='close', by='Hour', grid=True, figsize=(12,5))
        plt.show()

        df.trend.plot()
        plt.show()

        df.seasonal.plot()
        plt.show()

    plt.plot(df.close)
    plt.show()

def tensor_data(ticker=None, db_path=None, verbose=False):
    (df, train, test) = load_pure_data(ticker, db_path, split=0.8, verbose=False)
    (train_X, train_Y) = make_data_set(train, window_size=6, verbose=False)
    (test_X, test_Y) = make_data_set(test, window_size=6, verbose=False)
    if verbose is True: print('train data set:', train_X.shape, train_Y.shape)
    if verbose is True: print('test data set :', test_X.shape, test_Y.shape)
    (train_X, train_Y) = np_array2Tensor(train_X, train_Y, verbose=verbose)
    (test_X, test_Y)   = np_array2Tensor(test_X, test_Y, verbose=verbose)

    return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':
    db_path = "/root/work/coins/data/upbit/2022-07-12 17:00:00/"
    #ticker = 'KRW-WAXP'
    ticker = 'KRW-ETH'

    (df, train, test) = load_pure_data(ticker, db_path, split=0.8, verbose=True)
    (train_X, train_Y) = make_data_set(train, window_size=6, verbose=True)
    (test_X, test_Y) = make_data_set(test, window_size=6, verbose=False)
    print('train data set:', train_X.shape, train_Y.shape)
    print('test data set :', test_X.shape, test_Y.shape)
    (train_X, train_Y) = np_array2Tensor(train_X, train_Y, verbose=True)
    (test_X, test_Y)   = np_array2Tensor(test_X, test_Y, verbose=True)

    #correlation_analysis(verbose=True, Plot=False)