from preprocess.generateMap import ClusterGenerator
import matplotlib.pyplot as plt
from model.climateNet import ClimateNet
from dataReader.dataset import dataset
from torch.utils.data import  DataLoader, WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from sklearn import metrics

# Global Variable
 
SUBDATA_SIZE = 150
MODEL_NUM = 15
SPLIT_THRES = 0.9
LABEL_THRES = 0.5
APHI = 0.5
LR = 0.001

## helper Function for calculate acc
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

cg = ClusterGenerator(MODEL_NUM,"data_source/present_data.csv")
cg.loadData()
cg.kmeans()
cg.randomSample(decay=2,size=SUBDATA_SIZE)
# cg.showGrid() #Used for visualize the data cluster 




# data_frame = cg.meanShiftData
data_frame = cg.sampleData
print("Training dataFrame", data_frame.shape)
net_list = []
loss_list = []
for idx in range(MODEL_NUM):
    print()
    print("Working on model: ", idx)
    train_dataset = dataset(data_frame, idx, split_thres=SPLIT_THRES)
    val_dataset = dataset(data_frame, idx, split_thres=SPLIT_THRES, mode='val')
    ws = WeightedRandomSampler(torch.DoubleTensor(train_dataset.get_cls_label_weight()), len(train_dataset))
    trainloader = DataLoader(train_dataset,
                            sampler=ws, 
                            batch_size=16)
    valloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    net = ClimateNet(train_dataset.get_feature_len())
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    reg_loss_func = torch.nn.MSELoss()    
    # train_cls_loss_func = torch.nn.BCEWithLogitsLoss(weight=train_dataset.get_cls_weight())  # the target label is NOT an one-hotted
    # train_cls_loss_func = torch.nn.BCEWithLogitsLoss()
    # val_cls_loss_func = torch.nn.BCEWithLogitsLoss()  # the target label is NOT an one-hotted
    train_cls_loss_func = torch.nn.BCELoss()
    val_cls_loss_func = torch.nn.BCELoss()

    

    # print(net)
    cls_train_losses = []
    cls_train_accur = []
    cls_val_losses = []
    cls_val_accur = []

    reg_train_losses = []
    reg_val_losses = []

    total_losses = []
    print("start traning with dataset:",len(train_dataset),len(val_dataset))
    print()
    for i in range(100):
        optimizer.zero_grad()
        for j,(x, y_reg, y_cls) in enumerate(trainloader):
            #calculate output
            cls_train_output, reg_train_output = net(x)
            #calculate loss
            # cls_train_loss = cls_loss_func(cls_train_output, y_cls.reshape(-1,1))
            cls_train_loss = train_cls_loss_func(cls_train_output, y_cls.reshape(-1,1))
            reg_train_loss = reg_loss_func(reg_train_output, y_reg.reshape(-1,1))
        
            # loss = reg_loss_func(output,y )
            total_loss = APHI*cls_train_loss + (1-APHI)*reg_train_loss
            # total_loss = cls_train_loss
            #class accuracy
            
            cls_train_predicted, reg_train_predicted = net(train_dataset.get_x())
#             outputs = torch.sigmoid(cls_train_predicted).cpu()
            
#             print((sigmoid(cls_train_predicted.reshape(-1).detach().numpy()))>=LABEL_THRES)
            cls_train_acc = (cls_train_predicted.reshape(-1).detach().numpy().round()== train_dataset.get_y_cls().numpy()).mean()
            # cls_train_acc = metrics.balanced_accuracy_score(cls_train_predicted.reshape(-1).detach().numpy().round(), train_dataset.get_y_cls().numpy())



        for j_v,(x_v, y_v_reg, y_v_cls) in enumerate(valloader):

            cls_val_output, reg_val_output = net(x_v)

            #calculate loss
            cls_val_loss = val_cls_loss_func(cls_val_output, y_v_cls.reshape(-1,1))
            reg_val_loss = reg_loss_func(reg_val_output, y_v_reg.reshape(-1,1))
            #class accuracy

            cls_val_predicted, reg_val_predicted = net(val_dataset.get_x())
            cls_val_acc = (cls_val_predicted.reshape(-1).detach().numpy().round() == val_dataset.get_y_cls().numpy()).mean()

        
        cls_train_losses.append(cls_train_loss.data)
        cls_train_accur.append(cls_train_acc)
        cls_val_losses.append(cls_val_loss.data)
        cls_val_accur.append(cls_val_acc)
        reg_train_losses.append(reg_train_loss.data)
        reg_val_losses.append(reg_val_loss.data)
        
        
        if i%10 == 0:
            print("Epoch:",i)
            print("cls_train_loss: {:.5f} cls_val_loss: {:.5f} train_acc: {:.5f} val_acc: {:.5f}".format(cls_train_loss,cls_val_loss,cls_train_acc,cls_val_acc))
            print("reg_train_loss:{:.5f} reg_val_loss: {:.5f}".format(reg_train_loss,reg_val_loss))

        
        total_loss.backward()
        optimizer.step()
        
    loss_list.append([i,[cls_train_losses,cls_train_accur,reg_train_losses,reg_val_losses]])
    net_list.append([net,[cg.centers[idx]]])
    del(net)
    del(total_loss)
    del(optimizer)

plt.clf() 
for i in range(len(loss_list)):
    cls_train_losses_score = loss_list[i][1][0]
    plt.plot(cls_train_losses_score,label="Net:{},data#:{}".format(i,data_frame['model_{}'.format(i)].sum()))
    # plt.plot(cls_val_losses,label='val')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("fig/losses.png")

plt.clf()    
for i in range(len(loss_list)):
    cls_train_accur_score = loss_list[i][1][1]
    plt.plot(cls_train_accur_score,label="Net:{},data#:{}".format(i,data_frame['model_{}'.format(i)].sum()))
    # plt.plot(cls_val_losses,label='val')
    plt.title('accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("fig/accuracy.png")

plt.clf()  
for i in range(len(loss_list)):
    reg_train_losses = loss_list[i][1][2]
    plt.plot(reg_train_losses,label="Net:{},data#:{}".format(i,data_frame['model_{}'.format(i)].sum()))
    # plt.plot(cls_val_losses,label='val')
    plt.title('reg_train_losses vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("fig/reg_train_losses.png")




test_source = "data_source/test_data.csv"
test_dataset = dataset(pd.read_csv(test_source),0,0,mode='test',split=False)
testloader = DataLoader(test_dataset, batch_size=len(test_dataset)-1, shuffle=False)
cls_res = []
reg_res = []
for j_v,(x_v, y_v_reg, y_v_cls) in enumerate(testloader):
    for net_num in range(len(net_list)):
        net = net_list[i][0]
        la, lo = net_list[i][1][0]
        cls_val_output, reg_val_output = net(x_v)

        #calculate loss
        cls_val_loss = val_cls_loss_func(cls_val_output, y_v_cls.reshape(-1,1))
        reg_val_loss = reg_loss_func(reg_val_output, y_v_reg.reshape(-1,1))
        #class accuracy

        cls_predicted, reg_predicted = net(val_dataset.get_x())
        cls_res.append(cls_predicted.reshape(-1).detach().numpy().round())
        reg_res.append(reg_predicted.reshape(-1).detach().numpy())
        distence = np.sqrt((x_v[:,1]-la)**2+(x_v[:,2]-lo)**2)
        # cls_val_acc = (cls_val_predicted.reshape(-1).detach().numpy().round() == val_dataset.get_y_cls().numpy()).mean()