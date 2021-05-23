from torch.utils.data import Dataset
import torch
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import pandas as pd
np.random.seed(0)

class dataset(Dataset):
  def __init__(self ,data_frame, model_indx, mode='train'):
    self.x = None
    self.y_cls = None
    self.y_reg = None
    self.length = None
    self.data_np = None

    self.data_frame = data_frame
#     self.data_np = data_frame.to_numpy(dtype="float32")
    self.model_indx = model_indx
    self.mode = mode 

    # Initialize x, y_cls, y_reg
    self.split_data()

  def split_data(self):
#     print("Starting spliting data")
#     cls_data = self.data_np[self.data_np[:,-1]==self.model_indx] #Filtered by cluster
    if self.mode in ['train','val']:
      np_data = self.data_frame[self.data_frame['model_{}'.format(self.model_indx)]==1].to_numpy(dtype="float32")

    else:
      np_data = self.data_frame.to_numpy(dtype="float32")



    sc_x = MinMaxScaler()
    sc_reg = MinMaxScaler()
     
    x_full = np_data[:, 3:11].copy()             # coloumn name  [number,latitude,longitude,t_winter,t_spring,t_summer,t_fall,p_winter,p_spring,p_summer,p_fall,carb,veg,cluster], shape: (row, 8)
    y_full = np_data[:, 11:13].copy()            # coloumn name: [..... ,carb, veg, cluster], shape:(row, 3)
    self.centers = np_data[:,1:3]
    x = x_full.copy()
    y_reg = y_full[:,0].copy()
    y_cls = y_full[:,1].copy()

    x = sc_x.fit_transform(x)
    y_reg = sc_reg.fit_transform(y_reg.reshape(-1,1))
    y_cls -= 1
    y_reg = y_reg.reshape(-1)
                




    # if self.mode == 'train':
      # self.x = torch.tensor(x_train, dtype=torch.float32)
      # self.y_reg = torch.tensor(y_reg_train, dtype=torch.float32)
      # self.y_cls = torch.tensor(y_cls_train, dtype=torch.float32)
      # print(self.mode," dataset:", self.x.shape[0], ", positive:", self.y_cls.sum().numpy())


    # elif self.mode == 'val':
    #   self.x = torch.tensor(x_val, dtype=torch.float32)
    #   self.y_reg = torch.tensor(y_reg_val, dtype=torch.float32)
    #   self.y_cls = torch.tensor(y_cls_val, dtype=torch.float32)
    #   print(self.mode," dataset:", self.x.shape[0], ", positive:",self.y_cls.sum().numpy())

    self.x = torch.tensor(x, dtype=torch.float32)
    self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
    self.y_cls = torch.tensor(y_cls, dtype=torch.float32)

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
  
  def get_location(self):
    return self.centers

  def fit_transform(self, data_features, data_carbs, data_vegs):
    self.params = [np.std(data_features, axis=0, keepdims=True), np.mean(data_features, axis=0, keepdims=True), 
                    np.std(data_carbs, axis=0, keepdims=True), np.mean(data_carbs, axis=0, keepdims=True)]
    return self.transform(data_features, data_carbs, data_vegs)

  def transform(self, data_features, data_carbs, data_vegs):
    if self.params is None:
        raise ValueError('Not fit yet')
    return (data_features-self.params[1])/self.params[0], (data_carbs-self.params[3])/self.params[2], data_vegs-1

  def __getitem__(self,idx):
    return self.x[idx], self.y_reg[idx], self.y_cls[idx]

  def __len__(self):
    return self.length