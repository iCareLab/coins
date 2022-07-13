import os, sys
import torch

import lstm as network
import data_set as get

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from msglog import progress as bar

def network_parameter(device, X_train_tensors):
    """ 네트웨크 파라미터 구성하기 """
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
    print(X_train_tensors.shape[1])
    #lstm = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)
    #lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, _length).to(device)
    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, _length, device)

    loss_function = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # adam optimizer

    return num_epochs, lstm, loss_function, optimizer

def training(lstm, X_tensors, y_tensors, device):
    """ 학습하기 """
    bar.traning(progress=0, total=num_epochs)
    for epoch in range(num_epochs):
        outputs = lstm.forward(X_tensors.to(device)) #forward pass
 
        # obtain the loss function
        loss = loss_function(outputs, y_tensors.to(device))
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        loss.backward() #calculates the loss of the loss function
        optimizer.step() #improve from loss, i.e backprop

        if (epoch % 100 == 0) or (epoch+1 == num_epochs):
            bar.traning(epoch+1, num_epochs, epoch, loss.item())
    #print(colorama.Fore.RESET)


if __name__ == '__main__':
    """ 데이터셋 준비하기 """
    time, X_train, y_train, X_test, y_test = get.get_tensor_data('KRW-WAXP', '/root/work/coins/data/upbit/2022-07-12 17:00:00/', verbose=True)

    """ 네트웨크 파라미터 구성하기 """
    device = network.get_machine()
    (num_epochs, lstm, loss_function, optimizer) = network_parameter(device, X_train)

    """ 학습하기 """
    training(lstm, X_train, y_train, device)