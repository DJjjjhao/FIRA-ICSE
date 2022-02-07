from xmlrpc.client import FastParser
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from math import *

from combination_layer import CombinationLayer

def position_encoding(length, dmodel):
    pos = []
    for i in range(length):
        pos_cur = []
        for j in range(dmodel // 2):
            pos_cur.append(math.sin(i/(10000 ** (2 * j / dmodel))))
            pos_cur.append(math.cos(i/(10000 ** (2 * j / dmodel))))
        pos.append(pos_cur)
    pos_tensor = torch.tensor(pos)
    return pos_tensor

class Encoder(nn.Module):
    def __init__(self, args, pad_token_id):
        super(Encoder, self).__init__()

        self.dropout_rate = args.dropout_rate
        self.sou_len = args.sou_len
        self.att_len = args.att_len
        self.ast_change_len = args.ast_change_len
        self.sub_token_len = args.sub_token_len
        self.embedding_dim = args.embedding_dim
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim, padding_idx=pad_token_id)
        self.ast_change_embedding = nn.Embedding(
            num_embeddings=args.ast_change_vocab_size, embedding_dim=args.embedding_dim, padding_idx=pad_token_id)
        self.pos_encode = position_encoding(args.sou_len, self.embedding_dim)

        
        self.mark_embedding = nn.Embedding(num_embeddings=4, embedding_dim=args.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.embedding_dim, num_layers=3, batch_first=True)
        self.combination_list1 = nn.ModuleList([Combination(h=args.num_head, d_model=args.embedding_dim) for i in range(6)])
        self.combination_list2 = nn.ModuleList([Combination(h=args.num_head, d_model=args.embedding_dim) for i in range(6)])
        self.gcn_list = nn.ModuleList([GCN(args.embedding_dim, dropout_rate=0.2) for i in range(6)])

    def forward(self, input_token, sou_mask, attr, mark, ast_change, edge, sub_token):
        input_em = self.embedding(input_token) + self.pos_encode.to(input_token.device)
        # batch * len * emdbedding

        mark_em = self.mark_embedding(mark)
        ast_change_em = self.ast_change_embedding(ast_change)

        sub_token_em = self.embedding(sub_token)
        # batch
        for i in range(len(self.gcn_list)):
            
            input_em = self.combination_list2[i](input_em, input_em, mark_em)

            graph_em = torch.cat((input_em, sub_token_em, ast_change_em), dim = 1)
            # batch  * (code_len + sub_token_len + change_len) * embedding
            input_em, sub_token_em, ast_change_em= self.gcn_list[i](graph_em, edge, self.sou_len, self.sub_token_len, self.ast_change_len)

        return input_em, sub_token_em

class GCN(nn.Module):
    def __init__(self, dmodel, dropout_rate=0.1):
        super(GCN, self).__init__()

        self.dmodel = dmodel
        self.fc1 = nn.Linear(dmodel, dmodel)  
        self.fc2 = nn.Linear(dmodel, dmodel)  
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(dmodel)

    def forward(self, graph_em, edge, code_len, sub_token_len, ast_change_len):
        # graph_em: batch * len * embeding
        # edge: batch * len * len
        total_len = graph_em.size(1)
        x = self.fc1(graph_em)

        x = torch.bmm(edge.float(), x)  
        
        x = self.fc2(x)
        res = self.layernorm(self.dropout(x) + graph_em)

        assert res.size(1) == code_len + sub_token_len + ast_change_len
        return res[:,:code_len], res[:,code_len:code_len + sub_token_len], res[:, code_len + sub_token_len:]

class Decoder(nn.Module):
    def __init__(self, args, pad_token_id):
        super(Decoder, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim)
        self.pos_encode = position_encoding(args.tar_len, self.embedding_dim)

        self.tar_mask_pos = torch.ones(args.tar_len, args.tar_len)
        for i in range(args.tar_len):
            self.tar_mask_pos[i][i+1:] = 0

        self.attention_list = nn.ModuleList(
            [Attention(dmodel=args.embedding_dim, num_head=args.num_head) for i in range(6)])
        self.cross_attention_list = nn.ModuleList(
            [Attention(dmodel=args.embedding_dim, num_head=args.num_head) for i in range(6)])
        self.feed_forward_list = nn.ModuleList(
            [FeedForward(args.embedding_dim) for i in range(6)])

    def forward(self, output_token, input_em, sou_mask, tar_mask_pad):
        # input: batch * len 
        if torch.cuda.is_available():
            output_em = self.embedding(output_token) + self.pos_encode.cuda(output_token.device)
        else:
            output_em = self.embedding(output_token) + self.pos_encode

        # batch * len * emdbedding
        self.tar_mask_pos = self.tar_mask_pos.to(output_token.device)
        tar_mask = torch.logical_and(tar_mask_pad.unsqueeze(1).unsqueeze(1), self.tar_mask_pos.unsqueeze(0).unsqueeze(0))
        for i in range(len(self.attention_list)):
            output_em = self.attention_list[i](output_em, output_em, output_em, tar_mask)
            output_em = self.cross_attention_list[i](output_em, input_em, input_em, sou_mask)
            output_em = self.feed_forward_list[i](output_em)
        return output_em

class Attention(nn.Module):
    def __init__(self, dmodel, num_head, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.fc_q = nn.Linear(dmodel, dmodel)
        self.fc_k = nn.Linear(dmodel, dmodel)
        self.fc_v = nn.Linear(dmodel, dmodel)
        self.fc_o = nn.Linear(dmodel, dmodel)
        self.layernorm = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_head = num_head
        assert dmodel % self.num_head == 0
        self.dhead = dmodel // self.num_head

    def forward(self, query, key, value, mask):
        old_query = query
        batch_size = query.size(0)
        q_len = query.size(1)
        kv_len = key.size(1)
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        # batch * len * dmodel
        query = query.view(batch_size, q_len, self.num_head, self.dhead).transpose(1,2)
        key = key.view(batch_size, kv_len, self.num_head, self.dhead).transpose(1,2)
        value = value.view(batch_size, kv_len, self.num_head, self.dhead).transpose(1,2)
        weight = torch.matmul(query, key.transpose(-2,-1)) / sqrt(self.dhead) 
        # batch * num_head * len * len
        if len(mask.shape) < 4:
            mask = mask.unsqueeze(1).unsqueeze(1)
        weight = weight.masked_fill(mask == 0, -1e9)
        weight_softmax = F.softmax(weight, dim = -1)

        weighted_sum = torch.matmul(weight_softmax, value)
        # batch * num_head * len *embedding
        output = weighted_sum.transpose(1,2).contiguous().view(batch_size,q_len,self.num_head * self.dhead)
        output = self.fc_o(output)

        return self.layernorm(self.dropout(output) + old_query)

class FeedForward(nn.Module):
    def __init__(self, dmodel, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dmodel, 4 * dmodel)
        self.fc2 = nn.Linear(4 * dmodel, dmodel)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(dmodel)
    def forward(self, input_em):
        x = self.fc1(input_em)
        x = F.relu(x)
        x = self.fc2(x)
        return self.layernorm(self.dropout(x) + input_em)

class Combination(nn.Module):

    def __init__(self, h, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.combination = CombinationLayer()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        old_query = query

        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x = self.combination(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        output = self.output_linear(x)
        
        return self.layernorm(self.dropout(output) + old_query)
