import  pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import copy
from early_stopping import EarlyStopping
from data_process import Data,generate_data

path='../data/北京空气_2010.1.1-2014.12.31.csv'
data=generate_data(path)
c=Data(seq_len=24,fore_num=1,data=data)
xtrain,ytrain,xval,yval,xtest,ytest=c.process()


#标准化，不要axis=(0,1)，这样会造成数据的重复计算，造成分布偏移
xmean,xstd=xtrain.mean(axis=0),xtrain.std(axis=0)
ymean,ystd=ytrain.mean(axis=0),ytrain.std(axis=0)

xtrain,ytrain=(xtrain-xmean)/xstd,(ytrain-ymean)/ystd
xval,yval=(xval-xmean)/xstd,(yval-ymean)/ystd
xtest,ytest=(xtest-xmean)/xstd,(ytest-ymean)/ystd

xtrain,ytrain=torch.from_numpy(xtrain).type(torch.float32),torch.from_numpy(ytrain).type(torch.float32)
xval,yval=torch.from_numpy(xval).type(torch.float32),torch.from_numpy(yval).type(torch.float32)
xtest,ytest=torch.from_numpy(xtest).type(torch.float32),torch.from_numpy(ytest).type(torch.float32)

train_data=TensorDataset(xtrain,ytrain)
train_dl=DataLoader(train_data,batch_size=100,shuffle=True)

val_data=TensorDataset(xval,yval)
val_dl=DataLoader(val_data,batch_size=100,shuffle=True)

test_data=TensorDataset(xtest,ytest)
test_dl=DataLoader(test_data,batch_size=100,shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1 = torch.nn.LSTM(11, 50, 2, batch_first=True)
        # self.lstm2=torch.nn.LSTM(50,50,2,batch_first=True)
        self.drop = torch.nn.Dropout(0.5)
        self.linear1 = torch.nn.Linear(50, 1)
        # self.linear2=torch.nn.Linear(50,1)

    def forward(self, input):
        _, (x, _) = self.lstm1(input)
        x = self.drop(x[-1])
        out = self.linear1(x)
        #         x=self.drop(x)
        #         x=self.linear2(x)
        #out = torch.squeeze(x)  ##############注意，要不损失函数计算错误
        return out

if __name__=='__main__':
    model=Model()
    loss_fn=torch.nn.MSELoss(reduction='mean')
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                           patience=10, verbose=True, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    #best_model_weight = copy.deepcopy(model.state_dict())
    # best_test_loss = 1000000
    path= '../best_model.pth'
    es=EarlyStopping(path,delta=0)
    for i in range(1000):
        cnt = 0
        running_loss = 0
        model.train()  # 告诉模型是训练模式
        for x, y in train_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pre = model(x)
            y = y.to(torch.float32)
            x, y_pre = x.to(torch.float32), y_pre.to(torch.float32)
            loss = loss_fn(y_pre, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                running_loss += loss.item()
                cnt += 1

        epoch_loss = running_loss / cnt
        val_running_loss = 0
        cnt = 0
        model.eval()  # 告诉模型是预测模式
        with torch.no_grad():
            for x, y in val_dl:
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pre = model(x)
                y = y.to(torch.float32)
                x, y_pre = x.to(torch.float32), y_pre.to(torch.float32)
                loss = loss_fn(y_pre, y)
                val_running_loss += loss.item()
                cnt += 1

        val_epoch_loss = val_running_loss / cnt
        lr_reduce.step(val_epoch_loss)
        print('epoch:', i,
              '  loss:', round(epoch_loss, 3),
              '  val_loss:', round(val_epoch_loss, 3),
              '  lr', opt.param_groups[0]['lr'])

        es(val_epoch_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

    #预测
    m=model.load_state_dict(torch.load(path))
    plt.plot(model(xval).data.numpy()[-120:])
    plt.plot(yval.reshape(-1)[-120:], label='T')
    plt.legend()


