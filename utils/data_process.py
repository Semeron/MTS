import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import datetime


def generate_data(path='../data/北京空气_2010.1.1-2014.12.31.csv'):
    data = pd.read_csv(path)
    data = data.iloc[24:].fillna(method='ffill')
    data['tm'] = data.apply(lambda x: datetime.datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)
    data = data.drop(columns=['year', 'month', 'day', 'hour', 'No'])
    data = data.set_index('tm')
    data = data.join(pd.get_dummies(data['cbwd']))
    del data['cbwd']
    return data

class Data(object):
    def __init__(self,seq_len,fore_num,data,val_tatio=0.2,test_ratio=0.2):
        self.seq_len=seq_len
        self.fore_num=fore_num
        self.val_tatio=val_tatio
        self.test_ratio=test_ratio
        self.data=data

    def process(self):
        data = []
        for i in range(len(self.data) - self.seq_len - self.fore_num):
            data.append(self.data.values[i:i + self.seq_len + self.fore_num])
        data = np.array(data)
        #打乱数据，注意这里没有打乱时序的顺序，只是打乱的样本的顺序
        np.random.shuffle(data)
        X = data[:, :self.seq_len, :]
        Y = data[:, self.seq_len:, 0]
        #线划分数据集，再进行归一化，否则训练集数据的信息会干扰验证集、测试集
        val_split = int(len(data) *(1-self.val_tatio-self.test_ratio))
        test_split=int(len(data)*(1-self.val_tatio))
        xtrain,ytrain=X[:val_split],Y[:val_split]
        xval,yval=X[val_split:test_split],Y[val_split:test_split]
        xtest,ytest=X[test_split:],Y[test_split:]

        return xtrain,ytrain,xval,yval,xtest,ytest


if __name__=='__main__':
    data=generate_data()
    c=Data(24,2,data)
    xtrain,ytrain,xval,yval,xtest,ytest=c.process()
