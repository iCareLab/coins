
import torch
import torch.nn as nn
from torch.autograd import Variable 

def get_machine():
    """ 연산 machine선택: GPU or CPU """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
    print(device)
    #print(torch.cuda.get_device_name(0))
    return device

""" LSTM 네트워크 구성하기 """
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, device):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  #number of output classes
        self.num_layers = num_layers    #number of stacked lstm layer
        self.input_size = input_size    #input size: number of features
        self.hidden_size = hidden_size  #hidden state: number of features in hidden state 
        self.seq_length = seq_length    #sequence length
        self.device = device            #calculate on device
 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1
        self.fc   = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU() 

    def forward(self,x):
        self.hidden = (torch.zeros(self.num_layers, self.seq_length, self.hidden_size), \
                       torch.zeros(self.num_layers, self.seq_length, self.hidden_size))

        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state   
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state   

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
   
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
   
        return out