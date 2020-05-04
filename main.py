from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure, get_region_fmeasure
from model.seqlabel import SeqLabel
from utils.data import Data
from utils.functions import *
import os
import codecs

from model.optimizer import Optimizer

try:
    import cPickle as pickle
except ImportError:
    import pickle


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()

    data.label_alphabet.update_dict()

def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable[:,:,0].cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_entity(word_recover, pred_variable, gold_variable, pred_matrix, ans_matrix, label_alphabet):

    batch_size = pred_matrix.size(0)
    pred_matrix = pred_matrix[word_recover].cpu().data.numpy()
    ans_matrix = ans_matrix[word_recover].cpu().data.numpy()
    pred_tags = pred_variable[word_recover].cpu().data.numpy()
    gold_tags = gold_variable[word_recover].cpu().data.numpy()


    pred_entities = []
    real_entities = []

    for idx in range(batch_size):
        reg_pred_entity = matrix_to_entity(pred_matrix[idx], label_alphabet)
        seq_pred_entity = tag_to_entity(pred_tags[idx], label_alphabet) 

        pred_entity = seq_pred_entity.copy() 
        pred_entity.extend(reg_pred_entity)

        reg_real_entity = matrix_to_entity(ans_matrix[idx], label_alphabet)
        seq_real_entity = tag_to_entity(gold_tags[idx][:,0], label_alphabet)

        real_entity = seq_real_entity.copy()
        real_entity.extend(reg_real_entity)

        pred_entity = remove_dul(pred_entity)
        real_entity = remove_dul(real_entity)

        pred_entities.append(pred_entity)
        real_entities.append(real_entity) 

    return pred_entities, real_entities



def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, idx=0, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = [[] for i in range(data.max_depth)]
    gold_results = [[] for i in range(data.max_depth)]
    gold_words = []
    pred_entities = []
    real_entities = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    # batch_size = 10
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    fir_counts =  nex_counts = 0
    total_token = 0
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end] 
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, ans_matrix, wgt_matrix  = batchify_with_label(data, instance, data.HP_gpu, False, data.sentence_classification)
        
        total_token += sum(batch_wordlen)

        tag_seq, adj_matrix = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)

        p_tags, r_tags = recover_entity(batch_wordrecover, tag_seq, batch_label, adj_matrix, ans_matrix, data.label_alphabet)
        pred_entities.extend(p_tags) 
        real_entities.extend(r_tags)

    decode_time = time.time() - start_time
    speed = total_token/decode_time

    if not os.path.exists(data.output_dir):
        os.mkdir(data.output_dir)

    output_file = os.path.join(data.output_dir, "%s.eval.%i.output" % (name, idx))
    p, r, f = get_region_fmeasure(pred_entities, real_entities, data.label_alphabet, output_file)
   
    return speed, p, r, f, pred_results, pred_scores


def batchify_with_label(data, input_batch_list, gpu, if_train=True, sentence_classification=False):
    
    return batchify_sequence_labeling_with_label(data, input_batch_list, gpu, if_train)


def batchify_sequence_labeling_with_label(data, input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    adj_matrix = [sent[4] for sent in input_batch_list]
    wgt_matrix = [sent[5] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len, data.max_depth), requires_grad =  if_train).long()
    ans_seq_tensor = torch.zeros((batch_size, max_seq_len, max_seq_len)).long()
    wgt_seq_tensor = torch.FloatTensor(batch_size, max_seq_len, max_seq_len).fill_(-1)
   
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, label, seqlen, adj, wgt) in enumerate(zip(words, labels, word_seq_lengths, adj_matrix, wgt_matrix)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        ans_seq_tensor[idx, :seqlen, :seqlen] = torch.LongTensor(adj) 
        wgt_seq_tensor[idx, :seqlen, :seqlen] = torch.LongTensor(wgt)
       
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]

    ans_seq_tensor = ans_seq_tensor[word_perm_idx]
    wgt_seq_tensor = wgt_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        ans_seq_tensor = ans_seq_tensor.cuda()
        wgt_seq_tensor = wgt_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, ans_seq_tensor, wgt_seq_tensor


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)

    optimizer = Optimizer('sgd', 'adam', model, 'gcn', lr=data.HP_lr,
                      lr_gcn=data.HP_lr_gcn, momentum=data.HP_momentum, lr_decay=data.HP_lr_decay)
    best_dev = -10
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        sample_loss_flat = 0 
        sample_loss_graph = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, ans_matrix, wgt_matrix  = batchify_with_label(data, instance, data.HP_gpu, True, data.sentence_classification)
            instance_count += 1
            loss_flat, loss_graph, loss, tag_seq = model.calculate_loss(idx, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask, ans_matrix, wgt_matrix)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            sample_loss_flat += loss_flat.item()
            sample_loss_graph += loss_graph.item()
            total_loss += loss.item()
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss_flat: %.4f; loss_graph: %.4f; loss: %.4f; acc: %.4f"%(end, temp_cost, sample_loss_flat, sample_loss_graph, sample_loss, (right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
                sample_loss_flat = 0 
                sample_loss_graph = 0
            loss.backward()

            if data.HP_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), data.HP_clip)
                
            optimizer.step()
            model.zero_grad()

        optimizer.update(idx+1, batch_id+1, total_batch)

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss_flat: %.4f; loss_graph: %.4f; loss: %.4f; acc: %.4f"%(end, temp_cost, sample_loss_flat, sample_loss_graph, sample_loss, (right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, p, r, f, _,_ = evaluate(data, model, "dev", idx)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Test: time: %.2fs, speed: %.2fst/s; [p: %.4f, r: %.4f, f:  %.4f]"%(dev_cost, speed, p, r, f))

        if current_score > best_dev:
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir +'.'+ str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        # ## decode test
        speed, p, r, f, _,_ = evaluate(data, model, "test", idx)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; [p: %.4f, r: %.4f, f: %.4f]"%(test_cost, speed, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))

        gc.collect()


def load_model_test(data, name):
    print("Load Model from file: ", data.dset_dir)
    model = SeqLabel(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    start_time = time.time()
    speed, p, r, f, pred_results, pred_scores = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("Test: time: %.2fs, speed: %.2fst/s; [p: %.4f, r: %.4f, f:  %.4f]"%(time_cost, speed, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--wordemb',  help='Embedding for words', default='None')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 

    args = parser.parse_args()
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    if args.config == 'None':
        data.train_dir = args.train 
        data.dev_dir = args.dev 
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:",data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:",seed_num)
    else:
        data.read_config(args.config)
    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        if data.dset_dir is None:
            data_initialization(data)
            data.generate_instance('train')
            data.generate_instance('dev')
            data.generate_instance('test')
            data.build_pretrain_emb()
        else:
            data.load(data.dset_dir)
            data.read_config(args.config)
        train(data)
    elif status == 'test':
        print("MODEL: test")
        data.load(data.dset_dir)
        data.read_config(args.config)
        # data.show_data_summary()
        decode_results, pred_scores = load_model_test(data, 'test')
    else:
        print("Invalid argument! Please use valid arguments! (train/test)")

