import  pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
import  os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'utils'))
from utils.early_stopping import EarlyStopping
from utils.data_process import Data,generate_data
from model.lstnet_model import LSTNet
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error,r2_score
from data.data import troch_data

# data=pd.read_csv('/Users/semeron/Desktop/文件/工作/多指标预测/数据/multivariate-time-series-data-master/traffic/traffic.txt',header=None)
# data=pd.read_csv('/Users/semeron/Desktop/文件/工作/多指标预测/数据/multivariate-time-series-data-master/solar-energy/solar_energy.txt',header=None)
# data=pd.read_csv('/Users/semeron/Desktop/文件/工作/多指标预测/数据/multivariate-time-series-data-master/electricity/electricity.txt',header=None)
data=pd.read_csv('/Users/semeron/Desktop/文件/工作/多指标预测/数据/multivariate-time-series-data-master/exchange_rate/exchange_rate.txt',header=None)

# data=generate_data('/Users/semeron/Desktop/文件/工作/多指标预测/数据/PRSA_data_2010.1.1-2014.12.31.csv')
seq_len=30
fore_num=1
f_dim=8
ymean,ystd,train_dl,val_dl,xtest,ytest=troch_data(seq_len,fore_num,data,bsz=100,val_tatio=0.2,test_ratio=0.2)
model=LSTNet(seq_len=seq_len, pre_len=fore_num,f_dim=f_dim,outchannel_in_covn=120,k_size_in_conv=3,h_dim_lstm=110,layer_lstm=1,ar=24)
loss_fn=torch.nn.MSELoss(reduction='mean')
opt=torch.optim.Adam(model.parameters(),lr=0.001)
lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=10, verbose=True, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

model_path= '../best_model/lstnet_model.pth'
es=EarlyStopping(model_path,delta=0)
train_loss=[]
test_loss=[]
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
    train_loss.append(epoch_loss)
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
    test_loss.append(val_epoch_loss)
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
model.load_state_dict(torch.load(model_path))
test_pre=model(xtest)
test_pre=test_pre.data.numpy()*ystd+ymean
ytest=ytest.data.numpy()*ystd+ymean
mse = mean_squared_error(ytest, test_pre)
mae = mean_absolute_error(ytest, test_pre)
mape = mean_absolute_percentage_error(ytest, test_pre)
r2=r2_score(ytest, test_pre)

pre=np.concatenate([test_pre[0,:].reshape(-1),test_pre[1:,-1].reshape(-1)])
true=np.concatenate([ytest[0,:].reshape(-1),ytest[1:,-1].reshape(-1)])
fig,ax=plt.subplots(2,1,figsize=(19,9))
ax[0].plot(np.arange(1,len(train_loss)+1),train_loss,c='b',label='train_loss')
ax[0].plot(np.arange(1,len(test_loss)+1),test_loss,c='r',label='val_loss')
ax[1].plot(pre, label='pre')
ax[1].plot(true, label='true')
ax[0].legend()
ax[1].legend()
plt.suptitle('lstnet')
plt.show()
print(f'mse:{mse},mae:{mae},mape:{mape},r2:{r2}')



