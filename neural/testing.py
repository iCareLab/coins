import os, sys
import datetime as dt
from turtle import color
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import lstm as network
import data_set as get

def load_model(PATH):
    ''' method #1 '''
    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    ''' method #2 '''
    #checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기
    #model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    return model

def estimate(model, device, ticker, db_path):
    """ 데이터 가져오기 """
    X_train, y_train, X_test, y_test = get.tensor_data(ticker, db_path, verbose=False)

    """ 예측하기 """
    predict = model(X_test.to(device))#forward pass
    predict = predict.data.detach().cpu().numpy() #numpy conversion
    actual  = y_test.data.numpy()

    X = torch.cat([X_test,X_train], dim = 0)
    Y = torch.cat([y_train,y_test], dim = 0)
    Y = Y.data.numpy()

    plt.plot(y_train.data.numpy())
    plt.plot(actual)
    plt.show()

    return actual, predict, X, Y

#def result_plot(dataY_plot, data_predict, trains):
def result_plot(actual, predict, X, Y):
    """ plot 하기 """
    plt.figure(figsize=(10,6)) #plotting
    #plt.axvline(x=trains, c='r', linestyle='--') #size of the training set

    plt.stem(actual,  label='Actual Data') #actual plot
    plt.plot(predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.grid(axis='x')
    plt.show() 
    plt.savefig("savefig.pdf")

if __name__ == '__main__':
    CURR_DIR = os.getcwd()
    os.chdir('/root/work/coins/neural/models')
    PATH = str(dt.date.today()) + '-'
    model = load_model(PATH)
    os.chdir(CURR_DIR)

    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 모델을 이용하여 test 구간의 data를 얻는다.
    ticker  = 'KRW-ETH'
    db_path = '/root/work/coins/data/upbit/2022-07-12 17:00:00/'
    device = network.get_machine()
    actual, predict, X, Y = estimate(model, device, ticker, db_path)

    # 그래프로 확인한다.
    result_plot(actual, predict, X, Y)