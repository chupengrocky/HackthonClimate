from torch.utils.data import Dataset
import torch
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import pandas as pd
np.random.seed(0)

class dataset(Dataset):
  def __init__(self ,data_frame, model_indx, split_thres=0.8, mode='train', split = True):
    self.x = None
    self.y_cls = None
    self.y_reg = None
    self.length = None
    self.data_np = None
    self.split = split
    
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
    if self.mode in ['train','val']:
      cls_data = self.data_frame[self.data_frame['model_{}'.format(self.model_indx)]==1].to_numpy(dtype="float32")
#     print()
#     print("cls shape:", cls_data.shape)

      np.random.shuffle(cls_data)
    else:
      cls_data = self.data_frame.to_numpy(dtype="float32")
    
    
    x_full = cls_data[:, 3:11]             # coloumn name  [number,latitude,longitude,t_winter,t_spring,t_summer,t_fall,p_winter,p_spring,p_summer,p_fall,carb,veg,cluster], shape: (row, 8)
    y_full = cls_data[:, 11:13]              # coloumn name: [..... ,carb, veg, cluster], shape:(row, 3)
    
    if self.split:
      split_indx = int(x_full.shape[0]*self.split_thres)
      ## Split traning data 
      x_train = x_full[:split_indx]
      y_reg_train = y_full[:split_indx, 0]
      y_cls_train = y_full[:split_indx, 1]
      
      ## Split validation data
      x_val = x_full[split_indx:]
      y_reg_val = y_full[split_indx:, 0]
      y_cls_val = y_full[split_indx:, 1]
    else:
      x_test = x_full[:]
      y_reg_test = y_full[:, 0]
      y_cls_test = y_full[:, 1]
    


    
    # y_train_z = np.zeros((y_cls_train.shape[0], int(y_cls_train.max()+1)))
    # y_train_z[np.arange(y_cls_train.shape[0]),y_cls_train.astype('int')] = 1
    # y_cls_train = y_train_z
    # y_val_z = np.zeros((y_cls_val.shape[0], int(y_cls_val.max()+1)))
    # y_val_z[np.arange(y_cls_val.shape[0]),y_cls_val.astype('int')] = 1
    # y_cls_val = y_val_z
    # y_cls_train = enc.fit_transform(y_cls_train.reshape(-1, 1))
    # y_cls_val = enc.transform(y_cls_val.reshape(-1, 1))
    # y_cls_train = y_cls_train.reshape(-1,1)
    # y_cls_val = y_cls_val.reshape(-1,1)

    
    if self.mode in ['train','val']:
    # Normalization and Standarization
      sc_x = MinMaxScaler()
      sc_reg = MinMaxScaler()
      
      x_train = sc_x.fit_transform(x_train)
      x_val = sc_x.transform(x_val)
      
      y_cls_train -= 1
      y_cls_val -= 1
      y_reg_train = sc_reg.fit_transform(y_reg_train.reshape(-1, 1))
      y_reg_val = sc_reg.transform(y_reg_val.reshape(-1, 1))

      y_reg_train = y_reg_train.reshape(-1)
      y_reg_val = y_reg_val.reshape(-1)

    if self.mode == 'test':

      sc1 = MinMaxScaler()
      sc2 = MinMaxScaler()
      x_test = sc1.fit_transform(x_test.reshape(-1, 1))
      y_reg_test = sc2.fit_transform(y_reg_test.reshape(-1, 1))
      y_cls_test = y_cls_test-1


    if self.mode == 'train':
      self.x = torch.tensor(x_train, dtype=torch.float32)
      self.y_reg = torch.tensor(y_reg_train, dtype=torch.float32)
      self.y_cls = torch.tensor(y_cls_train, dtype=torch.float32)
      print(self.mode," dataset:", self.x.shape[0], ", positive:", self.y_cls.sum().numpy())


    elif self.mode == 'val':
      self.x = torch.tensor(x_val, dtype=torch.float32)
      self.y_reg = torch.tensor(y_reg_val, dtype=torch.float32)
      self.y_cls = torch.tensor(y_cls_val, dtype=torch.float32)
      print(self.mode," dataset:", self.x.shape[0], ", positive:",self.y_cls.sum().numpy())
    
    elif self.mode == 'test':
      self.x = torch.tensor(x_test, dtype=torch.float32)
      self.y_reg = torch.tensor(y_reg_test, dtype=torch.float32)
      self.y_cls = torch.tensor(y_cls_test, dtype=torch.float32)

    self.length = len(self.x)
    

  def get_cls_label_weight(self):
    class_counts = [int(len(self.y_cls)-self.y_cls.sum().numpy()),int(self.y_cls.sum().numpy())]
    class_weights = [len(self.y_cls)/class_counts[i] for i in range(len(np.unique(self.y_cls)))]
    weights = [class_weights[int(self.y_cls[i].numpy())] for i in range(len(self.y_cls))]
    # print(weights)
    return weights

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
    return self.x[idx], self.y_reg[idx], self.y_cls[idx]

  def __len__(self):
    return self.length