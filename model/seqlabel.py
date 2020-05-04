from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF
from .myfunctions import *

class SeqLabel(nn.Module):
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.max_depth = data.max_depth
        self.sch_k = data.sch_k
        self.bias=data.bias
        self.lambda1 = data.lambda1
        self.lambda2 = data.lambda2
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.label_alphabet = data.label_alphabet

        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        if self.gpu:
            self.word_hidden = self.word_hidden.cuda()
        
    def calculate_loss(self, idx, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, ans_matrix, wgt_matrix):
        out_flat1, feat_1p = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size, seq_len = word_inputs.size()

        wgt_matrix.masked_fill_(wgt_matrix==1, self.lambda1)
        wgt_matrix.masked_fill_(wgt_matrix==0, 1)
        wgt_matrix.masked_fill_(wgt_matrix==-1, 0)

        ls_flat1, tag_seq = self.decode_outermost(out_flat1, mask, batch_label[:,:,0]) 
  
        if self.sch_k > 0:
            sch_p = self.sch_k / (self.sch_k + np.exp(idx / self.sch_k)) 
            sch_tag_seq = torch.zeros_like(tag_seq)
            for i in range(batch_size):
                rd = np.random.random() 
                sch_tag_seq[i] = batch_label[i,:,0] if rd <= sch_p else tag_seq[i]
            graph_matrix = entity_to_graph(sch_tag_seq, self.label_alphabet, mask, bias=self.bias)
        else:
            graph_matrix = entity_to_graph(batch_label[:,:,0], self.label_alphabet, mask, bias=self.bias)

        out_graph = self.word_hidden.extract_inner(feat_1p, graph_matrix)
        ls_graph = self.decode_inner(out_graph, wgt_matrix, ans_matrix)
       
        out_flat2 = self.word_hidden.interactions(feat_1p, out_graph, word_seq_lengths) 
        ls_flat2, tag_seq = self.decode_outermost(out_flat2, mask, batch_label[:,:,0]) 
       
        ls_flat = ls_flat1 + ls_flat2
        total_loss = ls_flat + self.lambda2 * ls_graph
     
        if self.average_batch:
            total_loss = total_loss / batch_size
        return ls_flat, ls_graph, total_loss, tag_seq

    def decode_outermost(self, outs, mask, label=None):

        if self.use_crf:
            if label is not None:
                total_loss = self.crf.neg_log_likelihood_loss(outs, mask, label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            batch_size, seq_len = outs.size()[:2]
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            if label is not None:
                total_loss = loss_function(score, label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        if label is not None:
            return total_loss, tag_seq 

        return tag_seq

    def decode_inner(self, out_graph, wgt_matrix, ans_matrix):
        
        batch_size, seq_len, _, num_ne = out_graph.size()
        wgt_matrix = wgt_matrix.view(batch_size, seq_len, -1)
        
        # CrossEntropy
        loss_func = nn.CrossEntropyLoss(reduction='none')
        ls_graph = loss_func(out_graph.view((-1, num_ne)), ans_matrix.view((-1, ))).view(ans_matrix.size())
        ls_graph= (ls_graph*wgt_matrix.float()).sum()

        return ls_graph

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        out_flat1, feat_1p = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size, seq_len = word_inputs.size()[:2]

        tag_seq = self.decode_outermost(out_flat1, mask) 
        tag_seq = self.correct_predict(tag_seq, mask)
        graph_matrix = entity_to_graph(tag_seq, self.label_alphabet, mask, bias=self.bias)

        out_graph = self.word_hidden.extract_inner(feat_1p, graph_matrix)
        out_flat2 = self.word_hidden.interactions(feat_1p, out_graph, word_seq_lengths) 
        tag_seq = self.decode_outermost(out_flat2, mask) 
        _, pred_inner = torch.max(out_graph, 3)

        return tag_seq, pred_inner

    
    def correct_predict(self, predicts, mask, tagScheme='BIO'):
        """
        + predict: current predictions for one sequence
        + index  : track index of previous layer predictions
        Correct the prediction of the first word if it's
        illegal. e.g. IOOBIII->BOOBIII
        Illegal labels will disappear as the training
        process continues.
        """
        predicts = predicts * mask.long()
        predicts[predicts == 0] = 1
        if tagScheme == 'BIO':
            for i in range(predicts.size(0)):
                if predicts[i, 0] > 0 and predicts[i, 0] % 2 == 0:
                    predicts[i, 0] = predicts[i, 0] + 1
        return predicts
