from preprocess.generateMap import *
from model.climateNet import ClimateNet
from sklearn.preprocessing import StandardScaler
from dataReader.dataset import dataset
from torch.utils.data import  DataLoader
import torch
import numpy as np

cg = ClusterGenerator(5,"data_source/present_data.csv")
cg.loadData()
cg.kmeans()
# cg.showGrid()

SPLIT_THRES = 0.8

data_np = cg.clusterData.to_numpy(dtype="float32")
print(data_np.shape)
net_list = []
for idx in range(cg.centers.shape[0]):
    cls_data = data_np[data_np[:,-1]==idx] #Filtered by cluster
    np.random.shuffle(cls_data)
    x_full = cls_data[:, 3:-3]
    y_full = cls_data[:, -2]
    

    split_indx = int(x_full.shape[0]*SPLIT_THRES)
    x_train = x_full[:split_indx]
    y_train = y_full[:split_indx]
    x_val = x_full[split_indx:]
    y_val = y_full[split_indx:]

    print(x_train.shape,x_val.shape)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.transform(x_val)
    print(type(x_train),type(x_val))

    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataset = dataset(x_train, y_train)
    val_dataset = dataset(x_val, y_val)

    trainloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    valloader = DataLoader(val_dataset,batch_size=16,shuffle=False)
    print(len(train_dataset))


    net = ClimateNet(x_full.shape[1])
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
    class_loss_func = torch.nn.BCELoss()  # the target label is NOT an one-hotted

    

    # print(net)
    train_losses = []
    train_accur = []
    val_losses = []
    val_accur = []
    for i in range(200):
        for j,(x,y) in enumerate(trainloader):
            #calculate output
            output = net(x)
        
            #calculate loss
            train_loss = class_loss_func(output,y.reshape(-1,1))
        
            # loss = reg_loss_func(output,y )
        
            #class accuracy
            
            predicted = net(torch.tensor(x_train, dtype=torch.float32))
            train_acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()

            #backprop


        for j_v,(x_v,y_v) in enumerate(valloader):

          val_output = net(x_v)
      
          #calculate loss
          val_loss = class_loss_func(val_output,y_v.reshape(-1,1))

          #class accuracy
          
          predicted = net(torch.tensor(x_val, dtype=torch.float32))
          val_acc = (predicted.reshape(-1).detach().numpy().round() == y_val).mean()

        if i%20 == 0:
            train_losses.append(train_loss.data)
            train_accur.append(train_acc)
            val_losses.append(val_loss.data)
            val_accur.append(val_acc)
            # print("Net: {} epoch: {:<5} train_loss:{:5f} train_accuracy:{:5f}".format(idx, i,train_loss,train_acc))
                # val_losses.append(val_loss.data)
                # val_accur.append(val_acc)
            print("epoch {:<5} train_loss:{:.5f} val_loss: {:.5f} train_accuracy:{:.5f} val_accuracy: {:.5f}".format(i,train_loss,val_loss,train_acc,val_acc))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    net_list.append([net,[train_losses,train_accur]])
