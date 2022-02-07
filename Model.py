from torch import nn
import torch.nn.functional as F
import torch
import math
from gnn_transformer import Encoder, Decoder

class CopyNet(nn.Module):
    def __init__(self, args):
        super(CopyNet, self).__init__()
        self.embedding_size = args.embedding_dim
        self.LinearSource = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 2)
    def forward(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(F.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        prob = F.softmax(self.LinearProb(traget), dim=-1)
        return genP, prob



class TransModel(nn.Module):
    def __init__(self, args):
        super(TransModel, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.vocab_size = args.vocab_size
        self.sou_len = args.sou_len
        self.sub_token_len = args.sub_token_len
        
        self.encoder = Encoder(args, pad_token_id=0)
        self.decoder = Decoder(args, pad_token_id=0)
        self.out_fc = nn.Linear(args.embedding_dim, args.vocab_size)
        self.gate_fc = nn.Linear(args.embedding_dim, 1)
        self.copy_net = CopyNet(args)
        
    def forward(self, sou, tar, attr, mark, ast_change, edge, tar_label, sub_token, stage = 'train'):
        # source：batch * source_len
        # target: batch * target_len

        sou_mask = sou != 0
        tar_mask_pad = tar != 0
        sub_token_mask = sub_token != 0 

        sou_embedding, sub_token_embedding = self.encoder(sou, sou_mask, attr, mark, ast_change, edge, sub_token)

        sou_embedding = torch.cat((sou_embedding, sub_token_embedding), dim=1)
        # batch * (diff len + sub len) * embedding
        sou_mask = torch.cat((sou_mask, sub_token_mask), dim=1)
        
        tar_embedding = self.decoder(tar, sou_embedding, sou_mask, tar_mask_pad)
        # batch * tar_len * embedding
        tar_output_gen = self.out_fc(tar_embedding)
        # batch * tar_len * tar_vocab_size
        tar_output_gen = F.softmax(tar_output_gen, dim=-1)

        tar_output_copy, gate = self.copy_net(sou_embedding, tar_embedding)
        # batch * tar_len * (diff len + sub len)
        # batch * tar_len * 2
        tar_output_copy = torch.masked_fill(tar_output_copy, sou_mask.unsqueeze(1) == 0, -1e9)
        tar_output_copy = F.softmax(tar_output_copy, dim=-1)

        tar_output = torch.cat((gate[:,:,0].unsqueeze(-1) * tar_output_gen, gate[:,:,1].unsqueeze(-1) * tar_output_copy), dim=-1)
        # batch * tar_len * (vocab size + diff len + sub len)
        
        # print(torch.sum(tar_output,dim=-1))

        tar_output = torch.log(tar_output.clamp(min=1e-10, max=1)) 

        pads = torch.zeros(tar_label.size(0),1) 
        if torch.cuda.is_available():
            label = torch.cat([tar_label, pads.cuda(sou.device)], dim=-1)
        else:
            label = torch.cat([tar_label, pads], dim=-1)

        label = label[:,1:]
        label = label.long()
        mask = label != 0
        
        loss = F.nll_loss(tar_output.view(-1, self.vocab_size + self.sou_len + self.sub_token_len), label.contiguous().view(-1), reduction = 'none')
        loss = loss.masked_fill(mask.view(-1)==False, 0)
        if stage == 'train':
            return loss.sum(), mask.sum()
        elif stage == 'dev' or stage == 'test':
            return torch.argmax(tar_output, dim=-1)