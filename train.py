from preprocess.generateMap import *
from model.climateNet import ClimateNet
from dataReader.dataset import dataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
from sklearn import metrics

# Global Variable
MODEL_NUM = 2
SPLIT_THRES = 0.7
LABEL_THRES = 0.5
APHI = 0.5
BETA = 0.5


cg = ClusterGenerator(MODEL_NUM,"data_source/present_data.csv")
cg.loadData()
cg.kmeans()
# cg.showGrid() #Used for visualize the data cluster 



data_np = cg.meanShiftData.to_numpy(dtype="float32")
# print(data_np.shape)
net_list = []
for idx in range(cg.centers.shape[0]):

    print("Working on model: ", idx)
    train_dataset = dataset(data_np, idx)
    val_dataset = dataset(data_np, idx, mode='val')

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(len(train_dataset))


    net = ClimateNet(train_dataset.get_feature_len())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    reg_loss_func = torch.nn.MSELoss()    
    train_cls_loss_func = torch.nn.BCEWithLogitsLoss(weight=train_dataset.get_cls_weight())  # the target label is NOT an one-hotted
    val_cls_loss_func = torch.nn.BCEWithLogitsLoss(weight=val_dataset.get_cls_weight())  # the target label is NOT an one-hotted

    

    # print(net)
    cls_train_losses = []
    cls_train_accur = []
    cls_val_losses = []
    cls_val_accur = []

    reg_val_losses = []
    reg_val_losses = []

    total_losses = []
    for i in range(400):
        for j,(x, y_reg, y_cls) in enumerate(trainloader):
            #calculate output
            cls_train_output, reg_train_output = net(x)
        
            #calculate loss
            # cls_train_loss = cls_loss_func(cls_train_output, y_cls.reshape(-1,1))
            cls_train_loss = train_cls_loss_func(cls_train_output, y_cls)
            reg_train_loss = reg_loss_func(reg_train_output, y_reg.reshape(-1,1))
        
            # loss = reg_loss_func(output,y )
            total_loss = APHI*cls_train_loss + BETA*reg_train_loss
            #class accuracy
            
            cls_train_predicted, reg_train_predicted = net(train_dataset.get_x())
            cls_train_acc = ((cls_train_predicted.reshape(-1).detach().numpy()>LABEL_THRES) == train_dataset.get_y_cls().numpy()).mean()
            # cls_train_acc = metrics.balanced_accuracy_score((cls_train_predicted.reshape(-1).detach().numpy()>=LABEL_THRES).astype(float), train_dataset.get_y_cls().numpy())
            #backprop


        for j_v,(x_v, y_v_reg, y_v_cls) in enumerate(valloader):

            cls_val_output, reg_val_output = net(x_v)

            #calculate loss
            cls_val_loss = val_cls_loss_func(cls_val_output, y_v_cls.reshape(-1,1))
            reg_val_loss = reg_loss_func(reg_val_output, y_v_reg.reshape(-1,1))
            #class accuracy

            cls_val_predicted, reg_val_predicted = net(val_dataset.get_x())
            cls_val_acc = ((cls_val_predicted.reshape(-1).detach().numpy()>=LABEL_THRES).astype(float) == val_dataset.get_y_cls().numpy()).mean()

        
        cls_train_losses.append(cls_train_loss.data)
        cls_train_accur.append(cls_train_acc)
        cls_val_losses.append(cls_val_loss.data)
        cls_val_accur.append(cls_val_acc)
        if i%20 == 0:
            print("epoch {:<5} cls_train_loss:{:.5f} cls_val_loss: {:.5f} train_accuracy:{:.5f} val_accuracy: {:.5f}".format(i,cls_train_loss,cls_val_loss,cls_train_acc,cls_val_acc))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    net_list.append([net,[cls_train_losses,cls_train_accur]])

for i in range(len(net_list)):
    cls_train_losses = net_list[i][1][0]
    plt.plot(cls_train_losses,label="Net:{},data#:{}".format(i,data_np[data_np[:,-1]==i].shape[0]))
    # plt.plot(cls_val_losses,label='val')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("fig/losses.png")

plt.clf()    
for i in range(len(net_list)):
    cls_train_accur = net_list[i][1][1]
    plt.plot(cls_train_accur,label="Net:{},data#:{}".format(i,data_np[data_np[:,-1]==i].shape[0]))
    # plt.plot(cls_val_losses,label='val')
    plt.title('accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("fig/accuracy.png")