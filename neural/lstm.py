## %%
import numpy as np
import pandas as pd
#import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable 

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#TRAIN_SET = 5000

import colorama
def progress_bar(progress, total, epoch=0, loss=0.0, color=colorama.Fore.YELLOW):
	percent = 100 * (progress / float(total))
	bar = '#' * int(percent) + '-' * (100 - int(percent))
	print(color + f"\r|{bar}| {percent:.2f}%, Epoch:{epoch+1}, loss:{loss:1.5f}", end="\r")
	if float(progress) >= float(total):
	    print(colorama.Fore.GREEN + f"\r|{bar}| {percent:.2f}%, Epoch:{epoch+1}, loss:{loss:1.5f}", end="\r")
## %%

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit
def prepare_dataset(ticker = None):
    ## %%
    """ 1. 데이터셋 준비하기 """
    if ticker is None:
        db_path = "/root/work/eMo/data/upbit/2022-06-29 08:00:00/"
        df = pd.read_csv(db_path + 'KRW-WAXP.csv', index_col='Unnamed: 0', parse_dates=True)
        #df = df.drop(columns=['volume'])
    else:
        df = upbit.get_ohlcv(ticker, interval='minutes10')
    print(df)

    # %%
    """
    open 시가
    high 고가
    low 저가
    close 종가
    volume 거래량
    Adj Close 주식의 분할, 배당, 배분 등을 고려해 조정한 종가

    확실한건 거래량(Volume)은 데이터에서 제하는 것이 중요하고, 
    Y 데이터를 Adj Close로 정합니다. (종가로 해도 된다고 생각합니다.)
    """
    #X = df.drop(columns='volume')
    #y = df.iloc[:, 5:6]
    #df = df[::-1]
    X = df.drop(columns='volume')
    y = df.iloc[:, 3:4]
    print(X)
    print(y)

    ## %%
    """
    학습이 잘되기 위해 데이터 정규화 
    StandardScaler	각 특징의 평균을 0, 분산을 1이 되도록 변경
    MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 변경
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    ss = StandardScaler()
    mm = MinMaxScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y) 

    # Train Data
    #TRAIN_SET = len(df.index) - 5
    TRAIN_SET = int(len(df.index) * 0.8)
    X_train = X_ss[:TRAIN_SET, :]
    X_test = X_ss[TRAIN_SET:, :]

    # Test Data 
    """
    얼마나 예측데이터와 실제 데이터의 정확도를 확인하기 위해 
    from sklearn.metrics import accuracy_score
    를 통해 정확한 값으로 확인할 수 있다.
    """
    y_train = y_mm[:TRAIN_SET, :]
    y_test = y_mm[TRAIN_SET:, :] 
    #print(X_train)

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape) 
    ## %%
    """
    torch Variable에는 3개의 형태가 있다. 
    data, grad, grad_fn 한 번 구글에 찾아서 공부해보길 바랍니다. 
    """
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors  = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors  = Variable(torch.Tensor(y_test))

    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final  = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0],  1, X_test_tensors.shape[1] )) 
    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

    return df, X_train_tensors_final, y_train_tensors, mm, ss, TRAIN_SET


""" 2. 연산 machine선택: GPU or CPU """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
print(device)
#print(torch.cuda.get_device_name(0))

""" 3. LSTM 네트워크 구성하기 """
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU() 

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state   

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
   
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
   
        return out

def network_parameter(X_train_tensors_final):
    """ 4. 네트웨크 파라미터 구성하기 """
    num_epochs = 30000 #30000 #1000 epochs
    learning_rate = 0.00001 #0.001 lr

    input_size = 5 #number of features
    #hidden_size = 2 #number of features in hidden state
    hidden_size = 7 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers

    num_classes = 1 #number of output classes 
    #_length = X_train_tensors_final.shape[1]
    _length = 7
    print(_length)
    print(X_train_tensors_final.shape[1])
    #lstm = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, _length).to(device)



    loss_function = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # adam optimizer

    return num_epochs, lstm, loss_function, optimizer

def training(lstm):
    """ 5. 학습하기 """
    progress_bar(0, num_epochs)
    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final.to(device)) #forward pass
 
        # obtain the loss function
        loss = loss_function(outputs, y_train_tensors.to(device))
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        loss.backward() #calculates the loss of the loss function
        optimizer.step() #improve from loss, i.e backprop

        if (epoch % 100 == 0) or (epoch+1 == num_epochs):
            progress_bar(epoch+1, num_epochs, epoch, loss.item())
    print(colorama.Fore.RESET)

def estimate(ss, mm):
    """ 6. 예측하기 """
    df_X = df.drop(columns='volume')
    df_X_ss = ss.transform(df_X)
    df_y_mm = mm.transform(df.iloc[:, 3:4])
    #df_X_ss = ss.transform(df.drop(columns='volume'))
    #df_y_mm = mm.transform(df.iloc[:, 5:6])

    df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    #reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
    train_predict = lstm(df_X_ss.to(device))#forward pass
    data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)

    return dataY_plot, data_predict

#from sklearn.metrics import accuracy_score
#accuracy_score(dataY_plot, data_predict)

def result_plot(dataY_plot, data_predict, trains):
    """ 7. plot 하기 """
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=trains, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actual Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show() 
    plt.savefig("savefig.pdf")

if __name__ == '__main__':
    """ 1. 데이터셋 준비하기 """

    import data_set
    db_path = "/root/work/eMo/data/upbit/2022-07-01 09:00:00/"
    ticker = 'KRW-WAXP'
    time, X_train, y_train, X_test, y_test = data_set.get_tensor_data(ticker, db_path)
    print('Time duration: ', time.dtype, time.shape)
    print('Train Tensor : ', X_train.shape, y_train.shape)
    print('Test Tensor  : ', X_test.shape, y_test.shape)


    '''
    """ 2. 연산 machine선택: GPU or CPU """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
    print(device)

    """ 3. LSTM 네트워크 구성하기 """

    """ 4. 네트웨크 파라미터 구성하기 """
    (num_epochs, lstm, loss_function, optimizer) = network_parameter(X_train_tensors_final)

    """ 5. 학습하기 """
    training(lstm)

    """ 6. 예측하기 """
    (dataY_plot, data_predict) = estimate(ss, mm)

    """ 7. model save & restore 하기 """
    # Save Model
    #MFILE = "./LSTM_model.pth"
    #torch.save(lstm.eval.state_dict(), MFILE)

    # Call Model
    #model = lstm(data_dim, hidden_size, output_dim, 1, seq_length).to(device)
    #model.load_state_dict(torch.load(MFILE), strict=False)
    #model.eval()

    """ 8. plot 하기 """
    result_plot(dataY_plot, data_predict, trains)
    '''

#https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM
#https://blog.naver.com/PostView.nhn?blogId=na_young_1124&logNo=222281343807&parentCategoryNo=&categoryNo=33&viewDate=&isShowPopularPosts=true&from=search