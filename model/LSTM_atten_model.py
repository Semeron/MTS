import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class LSTM_Atten(nn.Module):
    """搭建Decoder结构"""

    def __init__(self, look_back, pre_len,f_dim,h_dim,layer):
        super(LSTM_Atten, self).__init__()
        self.lstm = nn.LSTM(input_size=f_dim,  # 1个输入特征
                            hidden_size=h_dim,  # 隐状态h扩展为为128维
                            num_layers=layer,  # 1层LSTM
                            batch_first=True,  # 输入结构为(batch_size, seq_len, feature_size). Default: False
                            )
        self.lstmcell = nn.LSTMCell(input_size=h_dim, hidden_size=h_dim)
        self.drop = nn.Dropout(0.5)  # 掉落率
        self.pre_len = pre_len
        self.h_dim = h_dim
        self.look_back = look_back
        self.fc1 = nn.Linear(self.h_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3=nn.Linear(h_dim*2,h_dim)  #解码器的全局信息的浓缩


        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        H, (h, c) = self.lstm(x)  # 编码
        h = h[-1,:,:]
        c = c[-1,:,:]
        h_in=h
        H_pre = torch.empty((h.shape[0], self.pre_len, self.h_dim * 2))
        for i in range(self.pre_len):  # 解码
            h_t, c_t = self.lstmcell(h_in, (h, c))  # 预测
            h_atten = self.Atten(H,h_t)  # 获取结合了注意力的隐状态
            H_pre[:, i, :] = h_atten  # 记录解码器每一步的隐状态
            h, c = h_t, c_t  # 将当前的隐状态与细胞状态记录用于下一个时间步
            h_in=self.fc3(h_atten)

        return self.fc2(self.drop(self.fc1(H_pre))).squeeze(2)

    def Atten(self, H,h):
        h=h.unsqueeze(1)
        atten = torch.matmul(h, H.transpose(1, 2)).transpose(1, 2)  # 注意力矩阵
        atten=F.softmax(atten,dim=1)
        atten_H = atten * H  # 带有注意力的历史隐状态
        atten_H = torch.sum(atten_H, dim=1).unsqueeze(1)  # 按时间维度降维
        return torch.cat((atten_H, h), 2).squeeze(1)
