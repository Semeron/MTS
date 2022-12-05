import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=6000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)    #64*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    #64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))    #256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  #(weizhi,1,d)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)   #64*1*512

    def forward(self, x):     #[seq,batch,d_model]
        return x + self.pe.permute(1,0,2)[:,:x.size(1), :]   #64*64*512

class TransAm(nn.Module):
    def __init__(self, seq_len,feature_size,out_size,d_model=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding=nn.Linear(feature_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model)          #50*512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model*seq_len, d_model)
        self.fc1=nn.Linear(d_model,d_model//2)
        self.fc2 = nn.Linear(d_model // 2,out_size)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        src=self.embedding(src)        #[seq,batch,d_model]
        src = self.pos_encoder(src)    #[seq,batch,d_model]
        output = self.transformer_encoder(src, self.src_mask,self.src_key_padding_mask)

        output=output.reshape(output.shape[0],-1)
        output = self.decoder(output)
        output=self.fc1(F.dropout(output))
        output = self.fc2(F.dropout(output))
        return output



