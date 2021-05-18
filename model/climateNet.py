
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)    # reproducible


class ClimateNet(torch.nn.Module):

  def __init__(self,input_shape):
    super(ClimateNet,self).__init__()
    self.fc1 = torch.nn.Linear(input_shape,32)
    # self.fc2 = nn.Linear(32,64)
    self.fc3 = torch.nn.Linear(32,1)
    # self.y_class = nn.Linear(32,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    # x = torch.relu(self.fc2(x))
    # x = torch.sigmoid(self.fc3(x))
    x = self.fc3(x)
    x = torch.sigmoid(x)
   
    return x


# net = ClimateNet(x.shape[1])     # define the network
# if torch.cuda.is_available():
#     print("using Gpu")
#     net = net.cuda()
# print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
# reg_loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
# class_loss_func = torch.nn.BCELoss()  # the target label is NOT an one-hotted