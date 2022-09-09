import os, sys
import datetime as dt
import pandas as pd
import numpy as np
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
    (df, train, test) = get.load_pure_data(ticker, db_path, split=0.8, verbose=False)
    X_train, y_train, X_test, y_test = get.tensor_data(ticker, db_path, verbose=False)
    data = pd.concat([train, test])
    data['predict'] = np.NaN
    print(data)

    """ 예측하기 """
    #X= torch.concat([X_train, X_test])
    #predict = model(X.to(device))#forward pass
    predict = model(X_test.to(device))#forward pass
    predict = predict.data.detach().cpu().numpy() #numpy conversion

    """ 정상 가격으로 환원 """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(test[['close']])
    predict = scaler.inverse_transform(predict)
    i = 0
    for predicted in np.array(predict[::-1]):
        i += 1
        #print(i, predicted, data.close.iloc[len(data.index)-i])
        data.predict.iloc[len(data.index)-i] = predicted

    print(data)

    return data, train.index[-1]

#def result_plot(dataY_plot, data_predict, trains):
def result_plot(result, train_amount):
    """ plot 하기 """
    #plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=train_amount, c='r', linestyle='--') #size of the training set
    #plt.stem(pd_result['close'],  label='Actual Data') #actual plot
    plt.plot(result['close'],  label='Actual Data') #actual plot
    plt.plot(result['predict'], label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.grid(axis='x')
    plt.show() 
    plt.savefig("savefig.pdf")

if __name__ == '__main__':
    CURR_DIR = os.getcwd()
    #os.chdir('/root/work/coins/neural/models')
    #os.chdir('/workspaces/coins/neural/models')
    #PATH = str(dt.date.today()) + '-'
    MODEL_PATH = CURR_DIR + '/neural/models/' + str(dt.date.today()) + '-'
    model = load_model(MODEL_PATH)

    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 모델을 이용하여 test 구간의 data를 얻는다.
    ticker  = 'KRW-BTC'
    #db_path = '/root/work/coins/data/upbit/2022-08-10/'
    db_path = '/workspaces/coins/data/upbit/2022-08-25/'
    DB_PATH = CURR_DIR + '/data/upbit/' + str(dt.date.today()) + '/'

    device = network.get_machine()
    #pd_result, trains = estimate(model, device, ticker, db_path)
    pd_result, trains = estimate(model, device, ticker, DB_PATH)

    # 그래프로 확인한다.
    result_plot(pd_result, trains)