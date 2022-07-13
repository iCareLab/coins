import datetime as dt
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import lstm as network
import data_set as get

def load_model(mfile):
    ## 저장된 전체 모델 불러오기
    model = torch.jit.load(mfile)
    model.eval()

    return model

def estimate(model, device, ticker, db_path):
    """ 데이터 가져오기 """
    X_train, y_train, X_test, y_test = get.tensor_data(ticker, db_path, False)
    X_data = torch.cat([X_test,X_train], dim = 0)

    print(X_train.shape, X_test.shape, X_data.shape)

    """ 예측하기 """
    predict = model(X_data.to(device))#forward pass
    data_predict = predict.data.detach().cpu().numpy() #numpy conversion
    dataY_plot = y_test.data.numpy()

    return dataY_plot, data_predict

#def result_plot(dataY_plot, data_predict, trains):
def result_plot(dataY_plot, data_predict):
    """ 7. plot 하기 """
    plt.figure(figsize=(10,6)) #plotting
    #plt.axvline(x=trains, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actual Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show() 
    plt.savefig("savefig.pdf")

if __name__ == '__main__':
    MFILE = "./models/" + str(dt.date.today()) + "-model_scripted.pt"
    model = load_model(MFILE)

    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 모델을 이용하여 test 구간의 data를 얻는다.
    ticker  = 'KRW-WAXP'
    db_path = '/root/work/coins/data/upbit/2022-07-12 17:00:00/'
    device = network.get_machine()
    Y, predict = estimate(model, device, ticker, db_path)

    # 그래프로 확인한다.
    result_plot(Y, predict)