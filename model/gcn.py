import torch.nn as nn
import torch
import math

class GCN(nn.Module):
    def __init__(self, hid_size=256):
        super(GCN, self).__init__()
        
        self.hid_size = hid_size
        
        self.W = nn.Parameter(torch.FloatTensor(self.hid_size, self.hid_size//2).cuda())
        self.b = nn.Parameter(torch.FloatTensor(self.hid_size//2, ).cuda())
        
        self.linear_gcn = nn.Linear(hid_size // 2 *2, hid_size//2)
        self.init()
    
    def init(self):
        stdv = 1/math.sqrt(self.hid_size//2)
        
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj, is_relu=True):
        inp = torch.matmul(inp, self.W)+self.b # [BS, SL, HS]
        out = torch.matmul(adj, inp) # adj [BS, SL, SL] 

        batch_size, seq_len, _ = inp.size()

        if len(adj.size()) > 3:
            out = self.linear_gcn(out.transpose(0, 1).contiguous().transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        if is_relu==True:
            out = nn.functional.relu(out)
        
        return out
