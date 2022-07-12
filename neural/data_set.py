import os, sys
from statistics import variance
from matplotlib.pyplot import viridis
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit

def get_tensor_data(ticker=None, db_path=None, verbose=False):
    """
    Getting Pure Data
    """
    if (ticker is None) or (db_path is None):
        print('Cannot Open db file')
        return None

    df = pd.read_csv(db_path + ticker + '.csv', index_col='Unnamed: 0', parse_dates=True)
    #print(df)

    # df : 'open', 'high', 'low', 'close', 'volume', 'value'
    X = df.drop(columns='volume')   # 'open', 'high', 'low', 'close', 'value'
    y = df[['close']]               # 'close'
    if verbose is True: print(X)
    if verbose is True: print(y)

    """
    Normalize of getted data
    """
    ss = StandardScaler()
    mm = MinMaxScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)
    print('X:', X_ss[:5], '\ny:', y_mm[:5])

    """
    Split Data to training & test data
    """
    SPLIT = int(len(df.index) * 0.8)
    X_train_matrix = X_ss[:SPLIT, :]
    y_train_matrix = y_mm[:SPLIT, :]
    X_test_matrix  = X_ss[SPLIT:, :]
    y_test_matrix  = y_mm[SPLIT:, :]
    if verbose is True: print('Training Matrix(2nd-dim): X - ', X_train_matrix.shape, ', y -', y_train_matrix.shape)
    if verbose is True: print('Train Data Set: ', X_train_matrix[:5], y_train_matrix[:5])

    """
    Transfer to tensor data format
        scalar = 0-dimention = 0-tensor         :                       : (1 2 3 4 5 6)
        vector = 1-dimention = array = 1-tensor : shape(x, )            : [1 2 3 4 5 6]
        matrix = 2-dimention = 2-tensor         : shape(x, y)           : [[1 2 3][4 5 6]]
        tensor = N-dimention = N-tensor         : shape(x, y, z, .... ) : [[[1 2][3 4][5 6]]]
    """
    X_train_tensor_variable = Variable(torch.Tensor(X_train_matrix))
    y_train_tensor_variable = Variable(torch.Tensor(X_train_matrix))
    X_test_tensor_variable = Variable(torch.Tensor(X_test_matrix))
    y_test_tensor_variable = Variable(torch.Tensor(X_test_matrix))
    if verbose is True: print('Training Matrix(2nd-dim): X -', X_train_tensor_variable.shape, \
                                                      ', y -', y_train_tensor_variable.shape)
    if verbose is True: print('Train Tensor Forms: ', X_train_tensor_variable[:5], y_train_tensor_variable[:5])

    """
    torch Variable : data, grad, grad_fn
    """
    X_train_tensor = torch.reshape(X_train_tensor_variable, (X_train_tensor_variable.shape[0], 1, X_train_tensor_variable.shape[1]))
    X_test_tensor  = torch.reshape(X_test_tensor_variable,  (X_test_tensor_variable.shape[0],  1, X_test_tensor_variable.shape[1] )) 
    if verbose is True: print('Training Tensor(3rd-dim): X -', X_train_tensor.shape, ', y -', y_train_matrix.shape)
    if verbose is True: print('Train Tensors: ', X_train_tensor[:5], y_train_matrix[:5])

    time_series = df.index
    #print(time_series)
    
    return (time_series, X_train_tensor, y_train_vector, X_test_tensor, y_test_vector)

if __name__ == '__main__':
    db_path = "../data/upbit/2022-07-12 17:00:00/"
    ticker = 'KRW-WAXP'
    time, X_train, y_train, X_test, y_test = get_tensor_data(ticker, db_path, verbose=False)
    #print('Time duration: ', time.dtype, time.shape)
    #print('Train Tensor : ', X_train.shape, y_train.shape)
    #print('Test Tensor  : ', X_test.shape, y_test.shape)