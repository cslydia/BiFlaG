from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntNet(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, cnn_layer, kernel_type, dropout, gpu):
        super(IntNet, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.gpu = gpu
        self.cnn_layer = cnn_layer
        self.kernel_type = kernel_type
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))

        self.init_char_cnn_3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.init_char_cnn_5 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2) 

        self.cnn_list = nn.ModuleList() 
        self.multi_cnn_list_3 = nn.ModuleList() 
        self.multi_cnn_list_5 = nn.ModuleList() 
        
        last_dim = embedding_dim * self.kernel_type
        post_embedd_dim = embedding_dim // 2
        for idx in range(int((self.cnn_layer - 1) / 2)):
            self.cnn_list.append(nn.Conv1d(last_dim, post_embedd_dim, kernel_size=1, padding=0))
            self.multi_cnn_list_3.append(nn.Conv1d(post_embedd_dim, post_embedd_dim, kernel_size=3, padding=1))
            self.multi_cnn_list_5.append(nn.Conv1d(post_embedd_dim, post_embedd_dim, kernel_size=5, padding=2))

            last_dim += post_embedd_dim * self.kernel_type 

        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.init_char_cnn_3 = self.init_char_cnn_3.cuda()
            self.init_char_cnn_5 = self.init_char_cnn_5.cuda()
            
            for idx in range(int((self.cnn_layer - 1) / 2)):
                self.cnn_list[idx] = self.cnn_list[idx].cuda()
                self.multi_cnn_list_3[idx] = self.multi_cnn_list_3[idx].cuda()
                self.multi_cnn_list_5[idx] = self.multi_cnn_list_5[idx].cuda()
                

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_last_hiddens(self, input, seq_lengths, word_batch_size, word_max_seq):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size, max_seq = input.size()

        activate_func = F.relu

        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2,1).contiguous()
        char_cnn_out1 = activate_func(self.init_char_cnn_3(char_embeds))
        char_cnn_out3 = activate_func(self.init_char_cnn_5(char_embeds))
       
        last_cnn_feature = torch.cat([char_cnn_out1,  char_cnn_out3], 1)  

        for idx in range(int((self.cnn_layer - 1) / 2)):
            cnn_feature = activate_func(self.cnn_list[idx](last_cnn_feature)) 
            cnn_feature_3 = activate_func(self.multi_cnn_list_3[idx](cnn_feature))
            cnn_feature_5 = activate_func(self.multi_cnn_list_5[idx](cnn_feature)) 

            cnn_feature = torch.cat([cnn_feature_3,  cnn_feature_5], 1) 
            cnn_feature = torch.cat([cnn_feature, last_cnn_feature], 1)
            last_cnn_feature = cnn_feature

        char_cnn_out = last_cnn_feature
        char_cnn_out_max = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(word_batch_size, word_max_seq, -1)

        return char_cnn_out_max.view(batch_size, -1)

    
    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)

