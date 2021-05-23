from preprocess.generateMap import ClusterGenerator
import matplotlib.pyplot as plt
from model.climateNet import ClimateNet
from dataReader.dataset import dataset
from torch.utils.data import  DataLoader, WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import os 

if not os.path.isdir('res/'):
    os.mkdir('res/')
if not os.path.isdir('fig/'):
    os.mkdir('fig/')


## Select cluster process 
def run(EPOCH,SUBDATA_SIZE,MODEL_NUM,SPLIT_THRES,LR):
    cg = ClusterGenerator(MODEL_NUM,"data_source/train.csv")
    cg.loadData()
    cg.kmeans()
    cg.randomSample(decay=2,size=SUBDATA_SIZE)
    # cg.generateStack()   

    # cg.showGrid() #Used for visualize the data cluster 



    # Uncommend this line if using the localize Sampling 
    # data_frame = cg.meanShiftData
     # Uncommend this line if using the Random Sampling 
    data_frame = cg.sampleData



    print("Training dataFrame", data_frame.shape)
    train_frame = data_frame.sample(frac = SPLIT_THRES)
    val_frame = data_frame.drop(train_frame.index)
    net_list = []
    stats_list = []


    ## Training process 
    for idx in range(MODEL_NUM):
        print()
        print("Working on model: ", idx)
        train_dataset = dataset(train_frame, idx, mode = 'train')
        val_dataset = dataset(val_frame, idx, mode='val')
        ws = WeightedRandomSampler(torch.DoubleTensor(train_dataset.get_cls_label_weight()), len(train_dataset))
        trainloader = DataLoader(train_dataset,
                                # sampler=ws, 
                                batch_size=32,
                                shuffle=True
                                )
        valloader = DataLoader(val_dataset, batch_size=32,shuffle=False)


        net = ClimateNet(train_dataset.get_feature_len())
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        reg_loss_func = torch.nn.MSELoss()    
        # train_cls_loss_func = torch.nn.BCEWithLogitsLoss(weight=train_dataset.get_cls_weight())  # the target label is NOT an one-hotted
        # train_cls_loss_func = torch.nn.BCEWithLogitsLoss()
        # val_cls_loss_func = torch.nn.BCEWithLogitsLoss()  # the target label is NOT an one-hotted
        cls_loss_func = torch.nn.BCELoss()

        

        # print(net)
        cls_train_losses = []
        cls_train_accur = []
        cls_val_losses = []
        cls_val_accur = []

        reg_train_losses = []
        reg_val_losses = []

        total_losses = []
        best_loss = np.inf
        best_acc = 0
        save_weight = None
        print("start traning with dataset:",len(train_dataset),len(val_dataset))
        print()
        for i in range(EPOCH):
            epoch_train_cls_loss = 0
            epoch_train_reg_loss = 0
            epoch_total_train = 0
            epoch_total_val = 0
            epoch_valid_cls_loss = 0
            epoch_valid_reg_loss = 0
            epoch_total_valid = 0
            for j,(x, y_reg, y_cls) in enumerate(trainloader):
                epoch_total_train += x.shape[0]
                #calculate output
                cls_train_output, reg_train_output = net(x)
                #calculate loss
                cls_train_loss = cls_loss_func(cls_train_output, y_cls.reshape(-1,1))
                reg_train_loss = reg_loss_func(reg_train_output, y_reg.reshape(-1,1))
                #class accuracy
                epoch_train_cls_loss += cls_train_loss*x.shape[0]
                epoch_train_reg_loss += reg_train_loss*x.shape[0]
                # loss = reg_loss_func(output,y )
                total_loss = 0.5*cls_train_loss + 0.5*reg_train_loss
                # total_loss = cls_train_loss
                #class accuracy
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
            cls_train_predicted, reg_train_predicted = net(train_dataset.get_x())
            cls_train_acc = (cls_train_predicted.reshape(-1).detach().numpy().round() == train_dataset.get_y_cls().numpy()).mean()
            # cls_train_acc = metrics.balanced_accuracy_score(cls_train_predicted.reshape(-1).detach().numpy().round(), train_dataset.get_y_cls().numpy())
            epoch_train_cls_loss /= epoch_total_train
            epoch_train_reg_loss /= epoch_total_train


            for j_v,(x_v, y_v_reg, y_v_cls) in enumerate(valloader):
                epoch_total_valid += x_v.shape[0]
                #calculate output
                cls_val_output, reg_val_output = net(x_v)
                #calculate loss
                cls_val_loss = cls_loss_func(cls_val_output, y_v_cls.reshape(-1,1))
                reg_val_loss = reg_loss_func(reg_val_output, y_v_reg.reshape(-1,1))
                #class accuracy
                epoch_valid_cls_loss += cls_val_loss*x_v.shape[0]
                epoch_valid_reg_loss += reg_val_loss*x_v.shape[0]

            cls_val_predicted, reg_val_predicted = net(val_dataset.get_x())
            cls_val_acc = (cls_val_predicted.reshape(-1).detach().numpy().round() == val_dataset.get_y_cls().numpy()).mean()

            epoch_valid_cls_loss /= epoch_total_valid
            epoch_valid_reg_loss /= epoch_total_valid
            
            # if cls_val_acc > best_acc:
            #     print("Update save Weight at EPOCH {}, val acc: {}".format(i,best_acc))
            #     best_acc = cls_val_acc
            #     save_weight = net.state_dict()
            if epoch_valid_cls_loss < best_loss:
                print("Update save Weight at EPOCH {}, val loss: {}".format(i,best_loss))
                best_loss = epoch_valid_cls_loss
                save_weight = net.state_dict()

            cls_train_losses.append(epoch_train_cls_loss.data)
            cls_train_accur.append(cls_train_acc)
            cls_val_losses.append(epoch_valid_cls_loss.data)
            cls_val_accur.append(cls_val_acc)
            reg_train_losses.append(epoch_train_reg_loss.data)
            reg_val_losses.append(reg_val_loss.data)
            
            
            if i%10 == 0:
                print("Epoch:",i)
                print("cls_train_loss: {:.5f} cls_val_loss: {:.5f} train_acc: {:.5f} val_acc: {:.5f}".format(epoch_train_cls_loss,
                                                                                                            epoch_valid_cls_loss,
                                                                                                            cls_train_acc,
                                                                                                            cls_val_acc))
                print("reg_train_loss:{:.5f} reg_val_loss: {:.5f}".format(epoch_train_reg_loss,
                                                                        epoch_valid_reg_loss))

            
            
            
        stats_list.append([idx,[cls_train_losses, 
                            cls_train_accur, 
                            cls_val_losses, 
                            cls_val_accur, 
                            reg_train_losses, 
                            reg_val_losses]])
        net_list.append([save_weight,[cg.centers[idx]]])
        del net
        del total_loss,cls_train_loss,reg_train_loss,cls_val_loss,
        del optimizer
    

    ## save the loss in each epoch, commend this part when doing grid search
    plt.clf() 
    for i in range(len(stats_list)):
        cls_train_losses_score = stats_list[i][1][0]
        cls_val_losses_score = stats_list[i][1][2]
        plt.plot(cls_train_losses_score,label="train")
        plt.plot(cls_val_losses_score,label="val")
        # plt.plot(cls_val_losses,label='val')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("fig/mode_{}_cls_losses.png".format(i))
        plt.clf()

        
    for i in range(len(stats_list)):
        cls_train_accur_score = stats_list[i][1][1]
        plt.plot(cls_train_accur_score,label="Net{}".format(i))
        # plt.plot(cls_val_losses,label='val')
        plt.title('accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("fig/accuracy.png")

    plt.clf()  
    for i in range(len(stats_list)):
        reg_train_losses = stats_list[i][1][4]
        reg_val_losses = stats_list[i][1][5]
        plt.plot(reg_train_losses,label="train")
        plt.plot(reg_val_losses,label="val")
        # plt.plot(cls_val_losses,label='val')
        plt.title('reg_train_losses vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("fig/mode_{}_reg_losses.png".format(i))
        plt.clf()

    

    ### Test process 
    test_source = "data_source/test.csv"
    test_dataset = dataset(pd.read_csv(test_source),0,mode='test')
    testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    cls_res = []
    reg_res = []
    distances = []
    mse = []
    for j_v,(x_t, y_t_reg, y_t_cls) in enumerate(testloader):
        for net_num in range(len(net_list)):
            net = ClimateNet(x_t.shape[1])
            net.load_state_dict(net_list[net_num][0])
            net.eval()
            la, lo = net_list[net_num][1][0]
            distances.append(np.sqrt((test_dataset.get_location()[:,0]-la)**2+(test_dataset.get_location()[:,1]-lo)**2))

            
            #class accuracy

            cls_predicted, reg_predicted = net(x_t)
            cls_res.append(cls_predicted.reshape(-1).detach().numpy().round())
            reg_res.append(reg_predicted.reshape(-1).detach().numpy())
            # cls_val_acc = (cls_val_predicted.reshape(-1).detach().numpy().round() == val_dataset.get_y_cls().numpy()).mean()

    cls_res = np.array(cls_res)  #[model, data]
    reg_res = np.array(reg_res)  #[model, data]
    distances = np.array(distances)  #[data, model]
    probability = 1./(distances) 
    probability = probability/np.sum(probability, axis=0, keepdims=True)
    res_combine = (probability * cls_res).sum(axis=0)
    reg_combine = (probability * reg_res).sum(axis=0)
    cls_score = (res_combine.round() == y_t_cls.reshape(-1).detach().numpy()).mean()
    reg_score = ((reg_combine - y_t_reg.reshape(-1).detach().numpy())**2).mean()
    # prob.append(probability)
    print("Clssification Accuracy:",cls_score,"Regression Mean Square Error:",reg_score)
    return(cls_score,reg_score)

if __name__ =='__main__':
    EPOCH = [100,200]
    SUBDATA_SIZE = [100,150,200]
    MODEL_NUM = [5,10,15,20]
    SPLIT_THRES = [0.7,0.8]
    LR = [0.01, 0.05, 0.001, 0.005]


    ind = 0
    result_list = []
    
    # for e in EPOCH:
    #     for s in SUBDATA_SIZE:
    #         for m in MODEL_NUM:
    #             for sp in SPLIT_THRES:
    #                 for l in LR:
    #                     with open('res/result.txt','a+') as f:
    #                         print("Working on Epoch:{}, Size:{}, Model#:{}, Split%:{}, LearningRate:{}".format(e,s,m,sp,l))
    #                         cls_score,reg_score = run(e,s,m,sp,l)
    #                         re = np.array([cls_score,reg_score,e,s,m,sp,l])
    #                         np.save('res/result_{}.npy'.format(ind), re)
    #                         f.write("Epoch:{}, Size:{}, Model#:{}, Split:{}, LearningRate:{}, result: cls: {:.5f}  reg: {:.5f} \n".format(e,s,m,sp,l,cls_score,reg_score))
    #                         print("Finish...")
    #                         ind+=1
    #                     f.close()
    run(150,200,10,0.8,0.01)
    
