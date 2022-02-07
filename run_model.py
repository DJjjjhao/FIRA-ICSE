import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
from torch.optim import Adam
import nltk.translate.bleu_score as bleu_score
import torch.nn as nn
import random
import time
import traceback
import json
import sys

from Model import TransModel
from Dataset import TransDataset


use_cuda = torch.cuda.is_available()

smooth_func = bleu_score.SmoothingFunction().method2

if use_cuda:
    device_ids = list(range(torch.cuda.device_count()))

class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]
args = DotDict({
    'sou_len':210,
    'tar_len':30,
    'att_len':25,
    'ast_change_len':280,
    'sub_token_len':160,
    'lr':1e-4,
    'dropout_rate':0.1,
    'num_head':8,
    'embedding_dim':256, 
    'batch_size':  170 * len(device_ids) if use_cuda else 170, 
    'test_batch_size': 20,
    'epoches':150,
    'beam_size':3,
    'vocab_size':0,
    'ast_change_vocab_size':0
})

vocab = json.load(open('DataSet/word_vocab.json'))
r_vocab = {}
for each in vocab:
    r_vocab[vocab[each]] = each

args.vocab_size = len(vocab)

ast_change_vocab = json.load(open('DataSet/ast_change_vocab.json'))
args.ast_change_vocab_size = len(ast_change_vocab)

var_maps = json.load(open("DataSet/variable.json"))
all_index = json.load(open('all_index'))

def seed_everything(seed=0):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_tensor(x):
    assert isinstance(x, torch.Tensor)
    if use_cuda:
        x = x.cuda(device = device_ids[0])
    return x

def convert_ids_to_tokens(ids, r_vocab):
        tokens = []
        for each_id in ids:
            tokens.append(r_vocab[each_id])
        return tokens

def train(model, train_loader, optimizer, epoch, best_bleu, dev_loader):
    model.train()
    total_data = 0
    total_loss = 0
    for idx, batch in enumerate(train_loader):
        
        if epoch >= 15 and idx % 10 == 0:
        # if idx % 10 == 0:
            cur_bleu, output_str = dev(model, dev_loader, epoch)
            open('OUTPUT/train_process','a').write('epoch: {} batch: {} dev bleu: {} is better: {}\n'.format(epoch, idx, cur_bleu, cur_bleu > best_bleu))

            if cur_bleu > best_bleu:
                best_bleu = cur_bleu
                torch.save(model.module.state_dict(),"best_model.pt")
                open('OUTPUT/dev_output', 'w').write(output_str)
            
            model.train()

        assert isinstance(batch, list)
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
        loss, mask = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], 'train')
        loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_data += len(batch[0])
        total_loss += loss.item()

        if idx % 10 == 0:
            print("epoch: %d batch: %d/%d  data: %d/%d loss: %.4f"%(epoch, idx, len(train_loader), total_data, len(train_loader.dataset), total_loss / 10))
            total_loss = 0        
    return best_bleu
def dev(model, val_loader, epoch):
    valid_index = all_index['valid']

    model.eval()
    output_str = ''
    bleus = 0
    bleus_batch = 0
    total_data = 0
    for idx, batch in enumerate(val_loader):
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
        
        with torch.no_grad():   
        
            output = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], 'dev')
            # batch * tar_len 
            output = output.cpu().numpy()
            # batch * tar_len
            whole_input = batch[0].cpu().numpy()
            # batch * sou_len
            sub_input = batch[7].cpu().numpy()
            # batch * sub_token_len
        
            for i in range(len(output)):
                cur_idx = total_data + i 
                cur_var_map = var_maps[valid_index[cur_idx]]
                cur_r_var_map = {}
                for each in cur_var_map:
                    cur_r_var_map[cur_var_map[each]] = each
                
                each_whole_input = whole_input[i].tolist()
                each_sub_input = sub_input[i].tolist()
                each_sen = output[i].tolist()
                if vocab['<eos>'] in each_sen:
                    each_sen = each_sen[:each_sen.index(vocab['<eos>'])] 

                for t in range(len(each_sen)):
                    if each_sen[t] >= args.vocab_size + args.sou_len:
                        each_sen[t] = each_sub_input[each_sen[t] - args.vocab_size - args.sou_len]
                    elif each_sen[t] >= args.vocab_size:
                        each_sen[t] = each_whole_input[each_sen[t] - args.vocab_size]

                each_sen_split = convert_ids_to_tokens(each_sen, r_vocab)
                each_sen_string = ' '.join(each_sen_split)
                each_sen_string = each_sen_string.replace('<pad>',"").replace('<unkm>',"😅").strip()
                each_sen_split_new = each_sen_string.split()
                
                ref = list(batch[1][i].cpu().numpy())
                ref = ref[1:ref.index(vocab['<eos>'])]
                ref_split = convert_ids_to_tokens(ref, r_vocab)
                ref_string = ' '.join(ref_split)


                each_bleu = bleu_score.sentence_bleu([ref_split], each_sen_split_new, smoothing_function=smooth_func)
                bleus += each_bleu
                bleus_batch += each_bleu

                for j in range(len(each_sen_split_new)):
                    if each_sen_split_new[j] in cur_r_var_map:
                        each_sen_split_new[j] = cur_r_var_map[each_sen_split_new[j]]
                    
                output_str += ' '.join(each_sen_split_new) + ',' + str(each_bleu) + '\n'
        total_data += len(batch[0])
        if idx % 10 == 0:
            print("epoch: %d data: %d/%d bleu: %.4f"%(epoch, total_data, len(val_loader.dataset), bleus_batch / total_data))
    bleus /= len(val_loader.dataset)
    return bleus, output_str


def test(model, test_loader):
    test_index = all_index['test']
    
    model.eval()
    f = open("OUTPUT/output_fira",'w')
    bleus = 0
    total_data = 0

    all_over_num = 0  
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            for i in range(len(batch)):
                batch[i] = get_tensor(batch[i])

            sou_mask = batch[0] != vocab['<pad>']
            sub_token_mask = batch[7] != 0 
            sou_embedding, sub_token_embedding = model.encoder(batch[0], sou_mask, batch[2], batch[3], batch[4], batch[5], batch[7])
            sou_embedding = torch.cat((sou_embedding, sub_token_embedding), dim=1)
            # batch * (diff len + sub len) * embedding
            sou_mask = torch.cat((sou_mask, sub_token_mask), dim=1)

            batch_size = len(batch[0])
            gen = []  # batch * beam * seq len 
            prob = [] # batch * beam 

            for i in range(batch_size):
                gen_beam = []
                prob_beam = []
                for j in range(args.beam_size):
                    gen_beam.append([vocab['<start>']])
                    if j == 0:
                        prob_beam.append(1)
                    else:
                        prob_beam.append(0)
                gen.append(gen_beam)
                prob.append(prob_beam)
            
            for step in range(args.tar_len - 1):
                output_nexts = [] 
                cal_beam = 0
                uncomplete_beam_id = []
                for j in range(args.beam_size):  
                    test_batch = []  # batch * len 
                    test_prob = []  # batch 
                    batch_mask = [1] * batch_size  
                    for i in range(batch_size): 
                        cur = gen[i][j] 
                        if cur[-1] == vocab['<eos>']:
                            batch_mask[i] = 0

                        if len(cur) < args.tar_len:
                            cur = cur + [vocab['<pad>']] * (args.tar_len - len(cur))

                        test_batch.append(cur)
                        test_prob.append(prob[i][j])

                    if sum(batch_mask) == 0:
                        continue
                    else:
                        uncomplete_beam_id.append(j)
                        cal_beam += 1
                        
                    test_batch = torch.tensor(test_batch)
                    tar_mask_pad = test_batch != vocab['<pad>']
                    test_batch = get_tensor(test_batch)
                    tar_mask_pad = get_tensor(tar_mask_pad)
                    test_prob = get_tensor(torch.tensor(test_prob))
                    batch_mask = get_tensor(torch.tensor(batch_mask))
                    tar_embedding = model.decoder(test_batch, sou_embedding, sou_mask, tar_mask_pad)
                    output_gen = F.softmax(model.out_fc(tar_embedding), dim=-1)
                    
                    output_copy, gate = model.copy_net(sou_embedding, tar_embedding)
                    # batch * tar_len * (diff len + sub len)
                    # batch * tar_len * 2
                    output_copy = torch.masked_fill(output_copy, sou_mask.unsqueeze(1) == 0, -1e9)
                    output_copy = F.softmax(output_copy, dim=-1)

                    output = torch.cat((gate[:,:,0].unsqueeze(-1) * output_gen, gate[:,:,1].unsqueeze(-1) * output_copy), dim=-1)

                    output = output.detach()
                    # batch * tar len * (vocab size + diff len + sub len)
                    output_next = output[:,step,:]
                    # batch * (vocab size + diff len + sub len)
                    output_next = output_next * test_prob.unsqueeze(-1)
                    # batch * (vocab size + diff len + sub len)
                    output_next = output_next.masked_fill(batch_mask.unsqueeze(-1) == 0, -1)
                    output_nexts.append(output_next)

                if cal_beam == 0:
                    all_over_num += 1

                    break
                    
                output_combine = torch.cat(output_nexts, dim = -1)

                # batch * (vocab size + diff len + sub len) × beam size
                ends = []
                prob_ends = []
                for i in range(batch_size):
                    batch_end = []
                    batch_prob = []
                    for j in range(args.beam_size):
                        if gen[i][j][-1] == vocab['<eos>']:
                            batch_end.append(j)
                            batch_prob.append(prob[i][j])
                    if len(batch_prob) < args.beam_size:
                        batch_prob = batch_prob + [-1] * (args.beam_size - len(batch_prob))
                    ends.append(batch_end)
                    prob_ends.append(batch_prob)
                prob_ends = get_tensor(torch.tensor(prob_ends))  # batch * beam size
                output_combine = torch.cat([output_combine, prob_ends], dim = -1)
                # assert batch * (tar vocab size × beam size + beam size)
                total_len = args.vocab_size + args.sou_len + args.sub_token_len
                assert output_combine.size(1) == total_len * cal_beam + args.beam_size
                
               

                sort_prob, indices = torch.sort(output_combine, descending=True, dim=-1)
                sort_prob_k = sort_prob[:, :args.beam_size]
                indices_k = indices[:, :args.beam_size]
             
                which_beams = indices_k // total_len
                which_tokens = indices_k % total_len
                
                gen_old = gen
                prob_old = prob
                gen = []  # batch * beam * seq len
                prob = sort_prob_k.cpu().numpy() # batch * beam
                
                whole_input = batch[0].cpu().numpy()
                # batch * sou_len
                sub_input = batch[7].cpu().numpy()
                # batch * sub_token_len

                for i in range(batch_size):
                    each_whole_input = whole_input[i].tolist()
                    each_sub_input = sub_input[i].tolist()

                    gen_beam = []
                    for j in range(args.beam_size):
                        which_beam = which_beams[i][j].item()
                        which_token = which_tokens[i][j].item()
                        if which_beam == cal_beam:  
                            gen_beam.append(gen_old[i][ends[i][which_token]])
                            assert prob_old[i][ends[i][which_token]] == prob_ends[i][which_token].item()
                        else:
                            if which_token >= args.vocab_size + args.sou_len:
                                which_token = each_sub_input[which_token - args.vocab_size - args.sou_len]
                            elif which_token >= args.vocab_size:
                                which_token = each_whole_input[which_token - args.vocab_size]
                            gen_beam.append(gen_old[i][uncomplete_beam_id[which_beam]] + [which_token])
             
                    gen.append(gen_beam)

            bleus_batch = 0
            for i in range(batch_size):
                
                cur_idx = total_data + i
                cur_var_map = var_maps[test_index[cur_idx]]
                cur_r_var_map = {}
                for each in cur_var_map:
                    cur_r_var_map[cur_var_map[each]] = each

                index = np.argmax(prob[i])
                each_sen = gen[i][index]
                each_sen_split = convert_ids_to_tokens(each_sen, r_vocab)
                each_sen_string = ' '.join(each_sen_split)
                each_sen_string = each_sen_string.replace('<start>', "").replace('<eos>',"").replace('<pad>',"").replace('<unkm>',"😅").strip()
                each_sen_split_new = each_sen_string.split()

                ref = list(batch[1][i].cpu().numpy())
                ref = ref[1:ref.index(vocab['<eos>'])]
                ref_split = convert_ids_to_tokens(ref, r_vocab)
                ref_string = ' '.join(ref_split)


                each_bleu = bleu_score.sentence_bleu([ref_split], each_sen_split_new, smoothing_function=smooth_func)

                bleus += each_bleu  
                bleus_batch += each_bleu 
                for j in range(len(each_sen_split_new)):
                    if each_sen_split_new[j] in cur_r_var_map:
                        each_sen_split_new[j] = cur_r_var_map[each_sen_split_new[j]]

                f.write(' '.join(each_sen_split_new) + '\n')
            f.flush()
            total_data += len(batch[0])
            print("data: %d/%d bleu: %f"%(total_data, len(test_loader.dataset), bleus_batch / len(batch[0])))
    
    
    f.close()
    bleus /= len(test_loader.dataset)
    print("early over / all batch: %d / %d"%(all_over_num, len(test_loader)))
    
def main_train():
    train_set = TransDataset(args, 'train')
    dev_set = TransDataset(args, 'valid')
    test_set = TransDataset(args, 'test')

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)
    model = TransModel(args)
    # if os.path.exists("best_model.pt"):
    #     model.load_state_dict(torch.load("best_model.pt"))
    if use_cuda:
        model = nn.DataParallel(model, device_ids = device_ids)
        model = model.cuda(device_ids[0])
    best_bleu = -1
    optimizer = Adam(model.parameters(), args.lr)
    
    for epoch in range(args.epoches):
        best_bleu = train(model, train_loader, optimizer, epoch, best_bleu, dev_loader)

def main_test():
    dev_set = TransDataset(args, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.test_batch_size)
        
    test_set = TransDataset(args, 'test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size)
    
    model = TransModel(args)
    if use_cuda:
        model.load_state_dict(torch.load("best_model.pt"))
        model = model.cuda(device_ids[0])
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load("best_model.pt", map_location=device))     
    test(model, test_loader)

if __name__ == '__main__':
    stage = str(sys.argv[1])
    seed_everything()
    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    if stage == 'train':
        main_train()
    elif stage== 'test':
        main_test()

