import sys
import numpy as np
import codecs
import torch

def tag_to_entity(tag_seq, label_alphabet):
    """
    Collect predicted tags(e.g. BIO)
    in order to get entities including nested ones
    """

    entities = []
    for idj, tag in enumerate(tag_seq):
        try:
            tag = label_alphabet.get_instance(tag)
            if tag.split('-', 1)[0] == 'B' or tag.split('-', 1)[0] == 'S':
                entities.append([idj, idj + 1, tag.split('-', 1)[1]])
            elif tag.split('-', 1)[0] == 'I' or tag.split('-', 1)[0] == 'E':
                entities[-1][1] += 1
        except:
            entities = []
            break

    return entities


def entity_to_graph(tag_seq, label_alphabet, mask=None, bias=1, gpu=True):
    batch_size, seq_len = tag_seq.size()
    bound_matrix = torch.FloatTensor(batch_size, seq_len, seq_len).fill_(0.)
    forw_matrix = torch.FloatTensor(batch_size, seq_len, seq_len).fill_(0.)

    if gpu:
        bound_matrix = bound_matrix.cuda() 
        forw_matrix = forw_matrix.cuda()

    for idx in range(batch_size):
        pred_entities = tag_to_entity(tag_seq[idx], label_alphabet) 
        
        for entity in pred_entities:    
            for item1 in range(entity[0], entity[1]-bias):
                for item2 in range(item1+bias, entity[1]):
                    bound_matrix[idx][item1][item2] = 1
                    
        length = sum(mask[idx]) if mask is not None else seq_len
        for jdx in range(length - 1):
            forw_matrix[idx][jdx][jdx + 1] = 1

    tot_matrix = torch.cat([bound_matrix.unsqueeze(0), forw_matrix.unsqueeze(0)], dim=0)

    return tot_matrix







