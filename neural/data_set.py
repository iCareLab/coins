import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit

def load_pure_data(ticker, db_path, split=0.8, verbose=False):
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
    full_data = df.drop(columns='value')   # 'open', 'high', 'low', 'close', 'volume'
    if verbose is True: print('Original data:\n', full_data)

    ''' Split Data Set to train data and test data '''
    SPLIT = int(len(df.index) * split)
    split_train = full_data[:SPLIT].iloc[:, :]
    split_test  = full_data[SPLIT:].iloc[:, :]
    if verbose is True: print('train set:', np.array(split_train.shape))
    if verbose is True: print('test set: ', np.array(split_test.shape))

    return (df, split_train, split_test)

def make_data_set(data, window_size=None, verbose=False):
    ''' Scaled Normalize '''
    if verbose is True: print('Train Data Set:')
    #X_scaled = (data.values)
    #Y_scaled = (data[['close']].values)    # 'close' column
    X_scaled = StandardScaler().fit_transform(data)
    Y_scaled = MinMaxScaler().fit_transform(data[['close']])    # 'close' column
    if verbose is True: print('Scaled X:\n', X_scaled[:5])
    if verbose is True: print('Scaled y:\n', Y_scaled[:5])

    ''' Make split by windows size '''
    X_data = []
    Y_data = []
    for i in range(window_size, len(X_scaled)-1):
        X_data.append(X_scaled[i-window_size:i, :])
        Y_data.append(Y_scaled[i:i+1])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    if verbose is True: print('X-data set:\n', X_data[:5], X_data.shape)
    if verbose is True: print('Y-data set:\n', Y_data[:5], Y_data.shape)

    return [X_data, Y_data]

def data2Tensor(data=None, scaler=None, verbose=False):
    ''' Normalize of Data '''
    if   scaler == 'standard': 
        print("Standard Scaling....")
        scaler = StandardScaler()
    elif scaler == 'min-max':
        print("Min-Max Scaling....")
        scaler = MinMaxScaler()
    else:
        print("scaler is one of 'standard' or 'min-max'")
        return 0
    
    scaled_data = scaler.fit_transform(data[1])
    if verbose is True: print('Scaled Data:\n', scaled_data[:5], '\n.....\n', scaled_data[-6:-1])


def tensor_data(ticker=None, db_path=None, verbose=False):
    (_df, X, y) = load_pure_data(ticker, db_path, verbose = True)

    """
    Normalize of getted data
    """
    ss = StandardScaler()
    mm = MinMaxScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)
    if verbose is True: print('----- X:\n', X_ss[:5], '\n----- y:\n', y_mm[:5])

    """
    Split Data to training & test data
    """
    SPLIT = int(len(X.index) * 0.8)
    X_train_matrix = X_ss[:SPLIT, :]; y_train_matrix = y_mm[:SPLIT, :]
    X_test_matrix  = X_ss[SPLIT:, :]; y_test_matrix  = y_mm[SPLIT:, :]
    if verbose is True: print('----- Training Matrix(2dim): X - ', X_train_matrix.shape, ', y -', y_train_matrix.shape)
    if verbose is True: print('----- Train Data Set: \n', X_train_matrix[:5], y_train_matrix[:5])

    """
    Transfer to tensor data format
        scalar = 0-dimention = 0-tensor         :                       : (1 2 3 4 5 6)
        vector = 1-dimention = array = 1-tensor : shape(x, )            : [1 2 3 4 5 6]
        matrix = 2-dimention = 2-tensor         : shape(x, y)           : [[1 2 3][4 5 6]]
        tensor = N-dimention = N-tensor         : shape(x, y, z, .... ) : [[[1 2][3 4][5 6]]]
    """
    X_train_tensor_variable = Variable(torch.Tensor(X_train_matrix))
    y_train_tensor_variable = Variable(torch.Tensor(y_train_matrix))
    X_test_tensor_variable = Variable(torch.Tensor(X_test_matrix))
    y_test_tensor_variable = Variable(torch.Tensor(y_test_matrix))
    if verbose is True: print('----- Training Tensor(2dim): X -', X_train_tensor_variable.shape, ', y -', y_train_tensor_variable.shape)
    if verbose is True: print('----- Train Tensor Variable sets: \n', X_train_tensor_variable[:5], '\n', y_train_tensor_variable[:5])
    if verbose is True: print('----- Testing Tensor(2dim): y -', X_test_tensor_variable.shape, ', y -', y_test_tensor_variable.shape)
    if verbose is True: print('----- Test Tensor Variable sets: \n', X_test_tensor_variable[:5], '\n', y_test_tensor_variable[:5])

    """
    torch Variable : data, grad, grad_fn
    """
    X_train_tensor = torch.reshape(X_train_tensor_variable, (X_train_tensor_variable.shape[0], 1, X_train_tensor_variable.shape[1]))
    y_train_tensor = y_train_tensor_variable
    X_test_tensor  = torch.reshape(X_test_tensor_variable,  (X_test_tensor_variable.shape[0],  1, X_test_tensor_variable.shape[1] )) 
    y_test_tensor  = y_test_tensor_variable
    if verbose is True: print('----- Training Reshape-X Tensor(3dim): X -', X_train_tensor.shape, ', y -', y_train_tensor.shape)
    if verbose is True: print('----- Train Reshape-X Tensors sets: \n', X_train_tensor[:5], '\n', y_train_tensor[:5])
    if verbose is True: print('----- Testing Reshape-X Tensor(3dim): X -', X_test_tensor.shape, ', y -', y_test_tensor.shape)
    if verbose is True: print('----- Test Reshape-X Tensors sets: \n', X_test_tensor[:5], '\n', y_test_tensor[:5])

    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

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

if __name__ == '__main__':
    db_path = "/root/work/coins/data/upbit/2022-07-12 17:00:00/"
    #ticker = 'KRW-WAXP'
    ticker = 'KRW-ELF'

    #X_train, y_train, X_test, y_test = tensor_data(ticker, db_path, verbose=False)
    #correlation_analysis(verbose=True, Plot=False)
    #train, test = gen_data_set(df=df, window_size=6, verbose=False)

    (df, train, test) = load_pure_data(ticker, db_path, split=0.8, verbose=True)
    (train_X, train_Y) = make_data_set(train, window_size=6, verbose=True)
    (test_X, test_Y) = make_data_set(test, window_size=6, verbose=True)
    #data2Tensor(data=test, scaler='min-max', verbose=True)