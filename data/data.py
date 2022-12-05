from utils.data_process import Data,generate_data
import  pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

def troch_data(seq_len,fore_num,data,bsz=100,val_tatio=0.2,test_ratio=0.2):
    c=Data(seq_len=seq_len,fore_num=fore_num,data=data,val_tatio=val_tatio,test_ratio=test_ratio)
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
    train_dl=DataLoader(train_data,batch_size=bsz,shuffle=True)

    val_data=TensorDataset(xval,yval)
    val_dl=DataLoader(val_data,batch_size=bsz,shuffle=True)

    return ymean,ystd,train_dl,val_dl,xtest,ytest