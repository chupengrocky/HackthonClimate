from preprocess.generateMap import *
from model.climateNet import ClimateNet
from sklearn.preprocessing import StandardScaler,normalize
from dataReader.dataset import dataset
from torch.utils.data import  DataLoader
import torch
import numpy as np

cg = ClusterGenerator(6,"data_source/present_data.csv")
cg.loadData()
cg.kmeans()
# cg.showGrid()

SPLIT_THRES = 0.7

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
    
#     y_train = (y_train-min(y_train))/(max(y_train)-min(y_train))
#     if len(np.unique(y_val))==1:
    y_train -= 1
    y_val -= 1
#     else:
#         y_val = (y_val-min(y_val))/(max(y_val)-min(y_val))
#     y_train = normalize(y_train.reshape(-1, 1))
#     y_val = normalize(y_val.reshape(-1, 1))

    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataset = dataset(x_train, y_train)
    val_dataset = dataset(x_val, y_val)

    trainloader = DataLoader(train_dataset,batch_size=16, shuffle=True)
    valloader = DataLoader(val_dataset,batch_size=16, shuffle=False)
    print(len(train_dataset))


    net = ClimateNet(x_full.shape[1])
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    class_loss_func = torch.nn.BCELoss()  # the target label is NOT an one-hotted

    

    # print(net)
    train_losses = []
    train_accur = []
    val_losses = []
    val_accur = []
    for i in range(300):
        for j,(x,y) in enumerate(trainloader):
            #calculate output
            class_output, reg_output = net(x)
        
            #calculate loss
            train_loss = class_loss_func(class_output,y.reshape(-1,1))
        
            # loss = reg_loss_func(output,y )
        
            #class accuracy
            
            class_predicted, reg_predicted = net(torch.tensor(x_train, dtype=torch.float32))
            train_acc = ((class_predicted.reshape(-1).detach().numpy()>0.5) == y_train).mean()

            #backprop


        for j_v,(x_v,y_v) in enumerate(valloader):

            class_val_output, reg_val_output = net(x_v)

            #calculate loss
            val_loss = class_loss_func(class_val_output,y_v.reshape(-1,1))

            #class accuracy

            class_val_predicted, reg_val_predicted = net(torch.tensor(x_val, dtype=torch.float32))
            val_acc = (class_val_predicted.reshape(-1).detach().numpy().round() == y_val).mean()

        
        train_losses.append(train_loss.data)
        train_accur.append(train_acc)
        val_losses.append(val_loss.data)
        val_accur.append(val_acc)
        if i%20 == 0:
            print("epoch {:<5} train_loss:{:.5f} val_loss: {:.5f} train_accuracy:{:.5f} val_accuracy: {:.5f}".format(i,train_loss,val_loss,train_acc,val_acc))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    net_list.append([net,[train_losses,train_accur]])

for i in range(len(net_list)):
    train_losses = net_list[i][1][0]
    plt.plot(train_losses,label="Net:{},data#:{}".format(i,data_np[data_np[:,-1]==i].shape[0]))
    # plt.plot(val_losses,label='val')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs * 10')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("fig/losses.png")

plt.clf()    
for i in range(len(net_list)):
    train_accur = net_list[i][1][1]
    plt.plot(train_accur,label="Net:{},data#:{}".format(i,data_np[data_np[:,-1]==i].shape[0]))
    # plt.plot(val_losses,label='val')
    plt.title('accuracy vs Epochs')
    plt.xlabel('Epochs * 10')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("fig/accuracy.png")