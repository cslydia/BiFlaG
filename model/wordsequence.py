from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .gcn import GCN
import math

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num
        self.HP_hidden_dim = data.HP_hidden_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        if data.char_feature_extractor == "IntNet":
            kernel_type = data.HP_intNet_kernel_type
            self.input_size = data.word_emb_dim + int( (data.HP_intNet_layer - 1) // 2 * (data.char_emb_dim // 2) * kernel_type + data.char_emb_dim * kernel_type)
     
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel-1)/2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        self.lstm_ne = nn.LSTM(lstm_hidden * 2, lstm_hidden, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden, data.label_alphabet_size)

        self.num_ne = data.num_ne
        self.gcn_layer = data.gcn_layer
        self.exp0_ne = nn.Linear(lstm_hidden*2, lstm_hidden)
        self.exp1_ne = nn.Linear(lstm_hidden*2, lstm_hidden)
        self.fc2rel = nn.Linear(lstm_hidden*2, self.num_ne)

        self.gcn_fw = nn.ModuleList([GCN(lstm_hidden*2) for _ in range(self.gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(lstm_hidden*2) for _ in range(self.gcn_layer)])
        self.gcn_fw_2 = nn.ModuleList([GCN(lstm_hidden*2) for _ in range(self.gcn_layer)])
        self.gcn_bw_2 = nn.ModuleList([GCN(lstm_hidden*2) for _ in range(self.gcn_layer)])

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            feature_out = lstm_out.transpose(1,0)
            
        feat_1p = self.droplstm(feature_out)
        outermost = self.extract_outermost(feat_1p, word_seq_lengths)

        return outermost, feat_1p


    def extract_outermost(self, feat, word_seq_lengths):
        """
            input:
                feat: (batch_size, seq_len, hidden_dim)
                word_seq_lengths: list of batch_size, (batch_size,1)
            output:
                outermost: (batch_size, seq_len, num_ne)
        """

        packed_words = pack_padded_sequence(feat, word_seq_lengths.cpu().numpy(), True)
        lstm_out, hidden = self.lstm_ne(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, True)
        outermost = self.hidden2tag(self.droplstm(lstm_out))
        return outermost


    def feat_expand(self, feat):
        """
            input:
                feat: (batch_size, seq_len, hidden_dim)
            output:
                expand: (batch_size, seq_len, seq_len, hidden_dim)
        """
        exp0 = F.relu(self.exp0_ne(feat))
        exp0 = self.droplstm(exp0)
        exp1 = F.relu(self.exp1_ne(feat))
        exp1 = self.droplstm(exp1)

        exp0 = exp0.view((exp0.shape[0], exp0.shape[1], 1, exp0.shape[2]))
        exp0 = exp0.expand((exp0.shape[0], exp0.shape[1], exp0.shape[1], exp0.shape[3]))
        exp1 = exp1.view((exp1.shape[0], 1, exp1.shape[1], exp1.shape[2]))
        exp1 = exp1.expand((exp1.shape[0], exp1.shape[2], exp1.shape[2], exp1.shape[3]))
        expand = torch.cat([exp0, exp1], dim=3)
        return expand


    def extract_inner(self, feat, matrix):
        """
            input:
                feat: (batch_size, seq_len, hidden_dim)
                matrix: (graph_num, batch_size, seq_len, seq_len)
            output:
                inners: (batch_size, seq_len, seq_len, num_ne)
        """
        expand = self.feat_expand(feat)
        matrix_bw = matrix.transpose(2,3)

        outs = feat
        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw[i](outs, matrix)
            out_bw = self.gcn_bw[i](outs, matrix_bw)    
            outs = self.droplstm(torch.cat([out_fw, out_bw], dim=-1)) 
        outs = self.feat_expand(outs)
        outs = expand + outs

        inners = self.fc2rel(outs)
        return inners


    def interactions(self, feat, matrix, word_seq_lengths):
        """
            input:
                feat: (batch_size, seq_len, hidden_dim)
                matrix: (batch_size, seq_len, seq_len, num_ne)
                word_seq_lengths: list of batch_size, (batch_size,1)
            output:
                outermost: (batch_size, seq_len, num_ne)
                
        """
        matrix = F.softmax(matrix, dim=3)
        matrix, pos_fw = torch.max(matrix, 3)

        matrix.masked_fill_(pos_fw ==0, 0)
        matrix_bw = matrix.transpose(1,2)

        outs = feat
        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw_2[i](outs, matrix)
            out_bw = self.gcn_bw_2[i](outs, matrix_bw)    
            outs = self.droplstm(torch.cat([out_fw, out_bw], dim=2))
        feat_2p = feat + outs
        outermost = self.extract_outermost(feat_2p, word_seq_lengths)
        
        return outermost


