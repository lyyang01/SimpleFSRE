import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTM, self).__init__() 
        #self.setup_seed(seed)
        self.forward_lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=1, bidirectional=False, batch_first=True)
        self.backward_lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=1, bidirectional=False, batch_first=True)
    
    def forward(self, x):
        batch_size,max_len,feat_dim = x.shape
        out1, (h1,c1) = self.forward_lstm(x)
        reverse_x = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            reverse_x[:,i,:] = x[:,max_len-1-i,:]
                
        out2, (h2,c2) = self.backward_lstm(reverse_x)

        output = torch.cat((out1, out2), 2)
        return output,(1,1)


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

from collections import OrderedDict
class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False, backend_model=None): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        ####TODO
        if backend_model == 'cp':
            ckpt = torch.load("./CP_model/CP")
            #import pdb
            #pdb.set_trace()
            temp = OrderedDict()
            ori_dict = self.bert.state_dict()
            for name, parameter in ckpt["bert-base"].items():
                if name in ori_dict:
                    temp[name] = parameter
            
            ori_dict.update(temp)
            self.bert.load_state_dict(ori_dict)
        
        #self.bert.load_state_dict(ckpt["bert-base"])
        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        ##
        '''
        self.linear_t = nn.Linear(768*2, 768)
        self.linear_h = nn.Linear(768*2, 768)
        '''
    
    def global_atten2(self, h_state, t_state, sequence_outputs):
        #the best model now, 2021/10/22, 86.12%
        t_temp0 = t_state.view(t_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        t_temp = torch.softmax(torch.matmul(sequence_outputs, t_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        t_temp = t_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        t_global_feature = torch.mean(t_temp * sequence_outputs, 1)
        t_state = torch.cat((t_state, t_global_feature), -1)
        
        h_temp0 = h_state.view(h_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        h_temp = torch.softmax(torch.matmul(sequence_outputs, h_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        h_temp = h_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        h_global_feature = torch.mean(h_temp * sequence_outputs, 1)
        h_state = torch.cat((h_state, h_global_feature), -1)
        return h_state, t_state
    
    def entity_atten(self, h_state, t_state, sequence_outputs, inputs):
        batch, dim = h_state.shape
        
        h_final = torch.zeros([batch, dim], dtype=torch.float32, device='cuda')
        t_final = torch.zeros([batch, dim], dtype=torch.float32, device='cuda')
        
        for idx in range(len(inputs['pos1'])):
            
            head_entity = sequence_outputs[idx, inputs['pos1'][idx]: inputs['pos1_end'][idx] + 1]
            tail_entity = sequence_outputs[idx, inputs['pos2'][idx]: inputs['pos2_end'][idx] + 1]
            n, m = head_entity.shape
            n2, m2 = tail_entity.shape
            #import pdb
            #pdb.set_trace()
            
            temp_h = torch.softmax(torch.matmul(head_entity, h_state[idx].view(-1, 1)), 0).expand(n, head_entity.shape[1])
            h_final[idx] = torch.mean(temp_h * head_entity, 0)
            
            temp_t = torch.softmax(torch.matmul(tail_entity, t_state[idx].view(-1, 1)), 0).expand(n2, tail_entity.shape[1])
            t_final[idx] = torch.mean(temp_t * tail_entity, 0)
        #import pdb
        #pdb.set_trace()    
        return h_final, t_final
    
    
    def forward(self, inputs, cat=True):
        if not self.cat_entity_rep:
            #import pdb
            #pdb.set_trace()
            #import pdb
            #pdb.set_trace()
            x = self.bert(inputs['word'], attention_mask=inputs['mask'])['pooler_output']
            #x = self.bert(inputs['word'], attention_mask=inputs['mask'])['last_hidden_state']
            
            ##insert local feature
            #local_final = self.windows_sequence(x, 5, self.bilstm)
            #x = torch.cat([local_final, x], dim=-1)
            #x = self.linear(x)
            #x = torch.mean(x, 1)
            
            return x
        else:
            ##this is concanate the start tokens of two entity mentions
            #import pdb
            #pdb.set_trace()
            
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            
            if cat:
            
                sequence_outputs = outputs['last_hidden_state'] # [20, 128, 768]
                tensor_range = torch.arange(inputs['word'].size()[0])  # inputs['word'].shape  [20, 128]
                h_state = outputs['last_hidden_state'][tensor_range, inputs["pos1"]] # h_state.shape [20, 768]
                t_state = outputs['last_hidden_state'][tensor_range, inputs["pos2"]] # [20, 768]
            
                batch_size, max_len, feat_dim = sequence_outputs.shape
                
                ###TODO delete the element in the middle, no effect
                '''
                mask_matrix = torch.ones([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
                temp = torch.zeros([feat_dim], dtype=torch.float32, device='cuda')
                for i in range(batch_size):
                    mask_matrix[i][inputs["pos1"][i]] = temp
                    mask_matrix[i][inputs["pos2"][i]] = temp
                sequence_outputs = sequence_outputs * mask_matrix
                '''
                #############################################
                
                #TODO, add two attention for obtaining the relation representation. the entity attention and the global attention.
                #the format of the attention is the matrix multiply with the softmax layer.
                
                #take this into outside
                '''
                h_state, t_state = self.global_atten2(h_state, t_state, sequence_outputs)
                h_state = self.linear_h(h_state)
                t_state = self.linear_t(t_state)
            
                '''
                #state = torch.cat((h_state, t_state), -1)
                
                #return state, outputs['last_hidden_state']
                
                
                return h_state, t_state, outputs['last_hidden_state']
            else:
                
                return outputs['pooler_output'], outputs['last_hidden_state']
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos1_end_index = 1
        
        pos2_in_index = 1
        pos2_end_index = 1
        
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)
                
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)
                
            cur_pos += 1
            ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        
        #import pdb
        #pdb.set_trace()
        
        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        
        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 #these positions are exactly the position of four special charaters
        
    ##TODO tokenize relation name and description
    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask    
        
        

class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        ##
        #self.bilstm = BiLSTM(768)
    
    
    def windows_sequence(self,sequence_output, windows, lstm_layer):
        batch_size, max_len, feat_dim = sequence_output.shape
        local_final = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            index_list = []
            for u in range(1, windows // 2 + 1):
                if i - u >= 0:
                    index_list.append(i - u)
                if i + u <= max_len - 1:
                    index_list.append(i + u)
            index_list.append(i)
            index_list.sort()
            temp = sequence_output[:, index_list, :]
            out,(h,b) = lstm_layer(temp)
            local_f = out[:, -1, :]
            local_final[:, i, :] = local_f
        return local_final
    
    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0] ## x.shape [100, 2]
        
        #import pdb
        #pdb.set_trace()
        ##
        #local_x = self.windows_sequence(x, 5, self.bilstm)
        #x = torch.cat((x, local_x), -1)
        
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens

class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.cat_entity_rep = cat_entity_rep

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index, mask


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens 

##TODO add relation encoder seperately
class BERTRelationEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        
        ##
        #self.bilstm = BiLSTM(768)
        #self.linear = nn.Linear(768*3, 768*2)
        #self.linear_t_gate = nn.Linear(768*2, 1)
        #self.linear_h_gate = nn.Linear(768*2, 1)
        
        #self.linear_t_gate2 = nn.Linear(768*2, 1)
        #self.linear_h_gate2 = nn.Linear(768*2, 1)
        #self.linear_gate = nn.Linear(768*3, 1)
        #self.linear = nn.Linear(768*2, 768)
    
    def forward(self, inputs):
    ##return relation representation directly
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])   
        final1 = outputs['pooler_output']
        final2 = outputs['last_hidden_state']
        
        return final1, final2
             

