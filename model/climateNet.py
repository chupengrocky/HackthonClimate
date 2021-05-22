
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)    # reproducible


class ClimateNet(torch.nn.Module):

    def __init__(self,input_shape):
        super(ClimateNet,self).__init__()
        self.fc1 = torch.nn.Linear(input_shape,16)
        self.relu1 = torch.nn.ReLU()
        self.dout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(16,32)
        self.fc3 = torch.nn.Linear(32,1)
        self.fc4 = torch.nn.Linear(32,1)

    def forward(self,x):
        x = self.fc1(x)
        # x = self.dout(x)
        x = self.fc2(x)
        x = self.relu1(x)
        # x = torch.relu(x)
        # x = torch.relu(self.fc2(x))
        x_class = torch.sigmoid(self.fc3(x))
        # x_class = self.fc3(x)
        x_reg = self.fc4(x)
        
   
        return x_class, x_reg


# net = ClimateNet(x.shape[1])     # define the network
# if torch.cuda.is_available():
#     print("using Gpu")
#     net = net.cuda()
# print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
# reg_loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
# class_loss_func = torch.nn.BCELoss()  # the target label is NOT an one-hotted