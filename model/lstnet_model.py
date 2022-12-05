import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNet(nn.Module):
    def __init__(self,seq_len, pre_len,f_dim,outchannel_in_covn,k_size_in_conv,h_dim_lstm,layer_lstm,ar):
        super().__init__()
        self.seq_len=seq_len
        self.pre_len=pre_len
        self.f_dim =f_dim
        self.outchannel_in_covn=outchannel_in_covn
        self.k_size_in_conv=k_size_in_conv
        self.h_dim_lstm=h_dim_lstm
        self.layer_lstm=layer_lstm
        self.ar=ar  #ar<seq_len

        self.conv1 = nn.Conv2d(1, self.outchannel_in_covn, kernel_size=(self.k_size_in_conv, self.f_dim),padding=(1,0))
        self.lstm=nn.LSTM(self.outchannel_in_covn,self.h_dim_lstm,num_layers=self.layer_lstm,batch_first=True)
        self.fc1=nn.Linear(self.h_dim_lstm*2,self.h_dim_lstm)
        self.fc2 = nn.Linear(self.h_dim_lstm * 2, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc4 = nn.Linear(self.f_dim,self.pre_len)
        self.lstmcell = nn.LSTMCell(input_size=self.h_dim_lstm, hidden_size=self.h_dim_lstm)
        self.ar_model=nn.Linear(self.ar,1)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.seq_len, self.f_dim)
        c = F.relu(self.conv1(c))
        c = F.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(0, 2, 1).contiguous()
        H, (h, c)= self.lstm(r)
        h = h[-1, :, :]
        c = c[-1, :, :]

        # atten
        h_in = h
        H_pre = torch.empty((h.shape[0], self.pre_len, self.h_dim_lstm * 2))
        for i in range(self.pre_len):  # 解码
            h_t, c_t = self.lstmcell(h_in, (h, c))  # 预测
            h_atten = self.Atten(H, h_t)  # 获取结合了注意力的隐状态
            H_pre[:, i, :] = h_atten  # 记录解码器每一步的隐状态
            h, c = h_t, c_t  # 将当前的隐状态与细胞状态记录用于下一个时间步
            h_in = self.fc1(h_atten)

        lin=self.fc2(H_pre)
        res=self.fc3(F.dropout(lin)).squeeze(2)

        # ar
        if (self.ar > 0):
            z = x[:, -self.ar:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.ar)
            z = self.ar_model(z)
            z = z.view(-1, self.f_dim)
            z=self.fc4(z)
            res =res+z
        return res

    def Atten(self, H, h):
        h = h.unsqueeze(1)
        atten = torch.matmul(h, H.transpose(1, 2)).transpose(1, 2)  # 注意力矩阵
        atten = F.softmax(atten, dim=1)
        atten_H = atten * H  # 带有注意力的历史隐状态
        atten_H = torch.sum(atten_H, dim=1).unsqueeze(1)  # 按时间维度降维
        return torch.cat((atten_H, h), 2).squeeze(1)
