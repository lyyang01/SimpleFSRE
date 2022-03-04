import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Siamese(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230, dropout=0, relation_encoder=None):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        if sentence_encoder.cat_entity_rep:
            self.hidden_size = hidden_size*3
            self.normalize = nn.LayerNorm(normalized_shape=hidden_size*2)
        
        else:
            self.hidden_size = hidden_size
            self.normalize = nn.LayerNorm(normalized_shape=hidden_size)
            
        self.drop = nn.Dropout(dropout)
        self.relation_encoder = relation_encoder
        #***********************************************
        self.linear_t = nn.Linear(hidden_size*2, hidden_size)
        self.linear_h = nn.Linear(hidden_size*2, hidden_size)
        #
        self.t_key = nn.Parameter(torch.randn(20, hidden_size, hidden_size))
        self.h_key = nn.Parameter(torch.randn(20, hidden_size, hidden_size))
        #************************************************
    
    
    def global_atten_entity(self, h_state, t_state, sequence_outputs, rel_vec=None):
        #the best model now, 2021/10/22, 86.12%
        t_temp0 = t_state.view(t_state.shape[0], 1, -1)
        
        #TODO use rel_vector to generate key vector [batch, dim]
        #[batch, max_len, dim] [batch, dim, dim]
        #*********************************
        if rel_vec is not None:
            m, n = rel_vec.shape
            rel_vec = rel_vec.view(m, 1, n).expand(m, sequence_outputs.shape[1], n)
            t_key = torch.bmm(rel_vec, self.t_key)
            h_key = torch.bmm(rel_vec, self.h_key)
            t_temp = torch.softmax(torch.tanh(torch.matmul(t_key, t_temp0.permute(0,2,1))), 1)#.squeeze() ##[20, 128, 1]
        #*********************************
        else:
            t_temp = torch.softmax(torch.tanh(torch.matmul(sequence_outputs, t_temp0.permute(0,2,1))), 1)#.squeeze() ##[20, 128, 1]
        
        t_temp = t_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        t_global_feature = torch.mean(t_temp * sequence_outputs, 1)
        t_state = torch.cat((t_state, t_global_feature), -1)
        t_state = self.linear_t(t_state)
        
        h_temp0 = h_state.view(h_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        if rel_vec is not None:
            h_temp = torch.softmax(torch.tanh(torch.matmul(h_key, h_temp0.permute(0,2,1))), 1)#.squeeze() ##[20, 128, 1]
        else:
            h_temp = torch.softmax(torch.tanh(torch.matmul(sequence_outputs, h_temp0.permute(0,2,1))), 1)#.squeeze() ##[20, 128, 1]
        h_temp = h_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        h_global_feature = torch.mean(h_temp * sequence_outputs, 1)
        h_state = torch.cat((h_state, h_global_feature), -1)
        h_state = self.linear_h(h_state)
        
        final = torch.cat((h_state, t_state), -1)
        
        #return h_state, t_state
        return final
    
    
    def global_atten_relation(self, rel_loc, sequence_outputs):
        #the best model now, 2021/10/22, 86.12%
        
        #import pdb
        #pdb.set_trace()
        
        t_temp0 = rel_loc.view(rel_loc.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        #import pdb
        #pdb.set_trace()
        t_temp = torch.softmax(torch.matmul(sequence_outputs, t_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        t_temp = t_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        t_global_feature = torch.mean(t_temp * sequence_outputs, 1)
        #t_state = torch.cat((t_state, t_global_feature), -1)
        '''
        h_temp0 = h_state.view(h_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        h_temp = torch.softmax(torch.matmul(sequence_outputs, h_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        h_temp = h_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        h_global_feature = torch.mean(h_temp * sequence_outputs, 1)
        h_state = torch.cat((h_state, h_global_feature), -1)
        '''
        return t_global_feature
    
    
    def forward(self, support, query, rel, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        
        ##get relation
        if self.relation_encoder:
            rel_gol, rel_loc = self.relation_encoder(rel)
        else:
            rel_gol, rel_loc = self.sentence_encoder(rel, cat=False)
        rel_loc = torch.mean(rel_loc, 1) #[B*N, D]
        
        
        #TODO
        #support,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        #query,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        support_h, support_t,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_h, query_t,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        
        support = self.global_atten_entity(support_h, support_t, s_loc, None)
        query = self.global_atten_entity(query_h, query_t, q_loc, None)
        
        # Layer Norm
        support = self.normalize(support)
        query = self.normalize(query)
        
        
        # Dropout ?
        support = self.drop(support)
        query = self.drop(query)
        
        #####TODO
        
        rel_loc_s = rel_loc.unsqueeze(1).expand(-1, K, -1).contiguous().view(s_loc.shape[0], -1)  # (B * N * K, D)
        rel_loc_q = rel_loc.unsqueeze(1).expand(-1, int(total_Q/N), -1).contiguous().view(q_loc.shape[0], -1)  # (B * N * K, D)
        
        #import pdb
        #pdb.set_trace()
        
        glo_s = self.global_atten_relation(rel_loc_s, s_loc)
        glo_q = self.global_atten_relation(rel_loc_q, q_loc)
        
        support = torch.cat((support, glo_s), -1)
        query = torch.cat((query, glo_q), -1)
        ####
        
        support = support.view(-1, N * K, self.hidden_size) # (B, N * K, D)
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)
        B = support.size(0) # Batch size
        support = support.unsqueeze(1) # (B, 1, N * K, D)
        query = query.unsqueeze(2) # (B, total_Q, 1, D)

        #  Dot production
        z = (support * query).sum(-1) # (B, total_Q, N * K)
        z = z.view(-1, total_Q, N, K) # (B, total_Q, N, K)
        
        #import pdb
        #pdb.set_trace()
        
        
        # Max combination
        logits = z.max(-1)[0] # (B, total_Q, N)

        # NA #ly actually, this is not done for none-of-above
        # Ignore NA policy
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)

        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred 
