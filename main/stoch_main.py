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

delay=1
seq_len=30

df=pd.read_csv("/Users/semeron/python/MTS/data/train.csv", parse_dates=["Date"], index_col='Date')
data=[]
for i in range(len(df)-seq_len-delay):
    data.append(df.iloc[i:i+seq_len+delay])
data=np.array([i.values for i in data])
x=data[:,:seq_len,:]
y=data[:,-1,0]
test_split=round(len(data)*0.20)
train_x=x[:-test_split]
train_y=y[:-test_split]
val_x=x[-test_split:]
val_y=y[-test_split:]

mean_x,std_x=train_x.mean(axis=(0,1)),train_x.std(axis=(0,1))
mean_y,std_y=train_y.mean(),train_y.std()
train_x=(train_x-mean_x)/std_x
val_x=(val_x-mean_x)/std_x
train_y=(train_y-mean_y)/std_y
val_y=(val_y-mean_y)/std_y

trainX,trainY=torch.from_numpy(train_x).type(torch.float32),torch.from_numpy(train_y).type(torch.float32)
valX,valY=torch.from_numpy(val_x).type(torch.float32),torch.from_numpy(val_y).type(torch.float32)
train_data=TensorDataset(trainX,trainY)
train_dl=DataLoader(train_data,batch_size=100,shuffle=True)

val_data=TensorDataset(valX,valY)
val_dl=DataLoader(val_data,batch_size=100,shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1=torch.nn.LSTM(5,50,2,batch_first=True)
        #self.lstm2=torch.nn.LSTM(50,50,2,batch_first=True)
        self.drop = torch.nn.Dropout(0.5)
        self.linear=torch.nn.Linear(50,1)
    def forward(self,input):
        _,(x,_)=self.lstm1(input)
        x=self.drop(x[-1])
        out=self.linear(x)
        out=torch.squeeze(out)  ##############注意，要不损失函数计算错误
        return out



if __name__=='__main__':
    model=Model()
    loss_fn=torch.nn.MSELoss(reduction='mean')
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                           patience=2, verbose=True, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    #best_model_weight = copy.deepcopy(model.state_dict())
    best_test_loss = 1000000
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
    m(valX)

