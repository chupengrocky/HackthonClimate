from torch.utils.data import Dataset
import torch
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class dataset(Dataset):
  def __init__(self ,data_frame, model_indx, mode='train', split_thres = 0.8):
    self.x = None
    self.y_cls = None
    self.y_reg = None
    self.length = None
    self.data_np = None
    
    self.data_frame = data_frame
#     self.data_np = data_frame.to_numpy(dtype="float32")
    
    self.model_indx = model_indx
    self.split_thres = split_thres
    self.mode = mode 

    # Initialize x, y_cls, y_reg
    self.split_data()

  def split_data(self):
#     print("Starting spliting data")
#     cls_data = self.data_np[self.data_np[:,-1]==self.model_indx] #Filtered by cluster
    cls_data = self.data_frame[self.data_frame['model_{}'.format(self.model_indx)]==1].to_numpy(dtype="float32")
    print(cls_data.shape)
    print('llll')
#     print()
#     print("cls shape:", cls_data.shape)
    np.random.shuffle(cls_data)
    
    
    x_full = cls_data[:, 3:11]             # coloumn name  [number,latitude,longitude,t_winter,t_spring,t_summer,t_fall,p_winter,p_spring,p_summer,p_fall,carb,veg,cluster], shape: (row, 8)
    y_full = cls_data[:, 11:13]              # coloumn name: [..... ,carb, veg, cluster], shape:(row, 3)
    

    split_indx = int(x_full.shape[0]*self.split_thres)
    x_train = x_full[:split_indx]
    y_reg_train = y_full[:split_indx, 0]
    y_cls_train = y_full[:split_indx, 1]
    

    x_val = x_full[split_indx:]
    y_reg_val = y_full[split_indx:, 0]
    y_cls_val = y_full[split_indx:, 1]
    


    # Normalization and Standarization

    sc_x = MinMaxScaler()
    sc_y = MinMaxScaler()
    x_train = sc_x.fit_transform(x_train)
    x_val = sc_x.transform(x_val)
    
    y_cls_train -= 1
    y_cls_val -= 1

    y_cls_train = y_cls_train.reshape(-1,1)
    y_cls_train = y_cls_train.reshape(-1,1)

    y_reg_train = sc_y.fit_transform(y_reg_train.reshape(-1, 1))
    y_reg_val = sc_y.transform(y_reg_val.reshape(-1, 1))

    y_reg_train = y_reg_train.reshape(-1)
    y_reg_val = y_reg_val.reshape(-1)

    if self.mode == 'train':
      self.x = torch.tensor(x_train, dtype=torch.float32)
      self.y_reg = torch.tensor(y_reg_train, dtype=torch.float32)
      self.y_cls = torch.tensor(y_cls_train, dtype=torch.float32)
      print(self.mode," dataset:", self.x.shape[0])
      print()


    elif self.mode == 'val':
      self.x = torch.tensor(x_val, dtype=torch.float32)
      self.y_reg = torch.tensor(y_reg_val, dtype=torch.float32)
      self.y_cls = torch.tensor(y_cls_val, dtype=torch.float32)
      print(self.mode," dataset:", self.x.shape[0])
      print()

    self.length = len(self.x)
    

  
  def get_cls_weight(self):
    return (len(self.y_cls)-self.y_cls.sum())/(self.y_cls.sum())
    
  def get_feature_len(self):
    return self.x.shape[1]

  def get_x(self):
    return self.x

  def get_y_cls(self):
    return self.y_cls

  def get_y_reg(self):
    return self.y_reg

  def __getitem__(self,idx):
    return self.x[idx],self.y_reg[idx],self.y_cls[idx]

  def __len__(self):
    return self.length