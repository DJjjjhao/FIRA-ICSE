import json, pickle
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import math
from tqdm import tqdm
from scipy import sparse
num_train = 75000
num_valid = 8000
num_test = 7661 

VOCAB_UPPER_CASE = json.load(open('VOCAB_UPPER_CASE'))
lemmatization = {"added": "add", "fixed": "fix", "removed": "remove", "adding": "add", "fixing": "fix", "removing": "remove"}

class TransDataset(Dataset):
    def __init__(self, args, data_name):
        super(TransDataset, self).__init__()

        self.data_name = data_name
        self.diff_len = args.sou_len
        self.msg_len = args.tar_len
        self.att_len = args.att_len
        self.ast_change_len = args.ast_change_len
        self.sub_token_len = args.sub_token_len
        
        self.graph_len = self.diff_len + self.sub_token_len + self.ast_change_len

        raw_diffs = json.load(open("DataSet/difftoken.json"))
        raw_diff_atts = json.load(open("DataSet/diffatt.json"))
        raw_diff_marks = json.load(open("DataSet/diffmark.json"))
        raw_msgs = json.load(open("DataSet/msg.json"))
        var_maps = json.load(open("DataSet/variable.json"))
        raw_changes = json.load(open('DataSet/change.json'))
        raw_asts = json.load(open('DataSet/ast.json'))
        raw_edge_change_codes = json.load(open('DataSet/edge_change_code.json'))
        raw_edge_change_asts = json.load(open('DataSet/edge_change_ast.json'))
        raw_edge_ast_codes = json.load(open('DataSet/edge_ast_code.json'))
        raw_edge_asts = json.load(open('DataSet/edge_ast.json'))

        assert len(raw_diffs) == len(raw_diff_atts) == len(raw_diff_marks) == len(raw_msgs) == len(var_maps) == len(raw_changes) == len(raw_edge_change_codes) == len(raw_edge_change_asts) == len(raw_asts) == len(raw_edge_ast_codes) == len(raw_edge_asts)

        self.vocab = json.load(open('DataSet/word_vocab.json'))

        if not os.path.exists('DataSet/ast_change_vocab.json'):
            ast_word = {}
            THRESHOLD = 1
            for i in range(len(raw_asts)):
                for word in raw_asts[i]:
                    word = word.lower()
                    if word not in ast_word:
                        ast_word[word] = 1
                    else:
                        ast_word[word] += 1
            ast_change_vocab = {'<pad>':0, 'update':1, 'delete':2, 'add':3, 'move':4, 'match':5}
            for word in ast_word:
                if ast_word[word] >= THRESHOLD:
                    ast_change_vocab[word] = len(ast_change_vocab)
            json.dump(ast_change_vocab, open('DataSet/ast_change_vocab.json', 'w'),indent=1)
    
        self.ast_change_vocab = json.load(open('DataSet/ast_change_vocab.json'))

        if not os.path.exists("processed_%s.pkl"%(data_name)):
            self.process_data(raw_diffs, raw_diff_atts, raw_diff_marks, raw_msgs, var_maps, raw_changes, raw_asts, raw_edge_change_codes, raw_edge_change_asts, raw_edge_ast_codes, raw_edge_asts)
        self.data = pickle.load(open("processed_%s.pkl"%(data_name),'rb'))
        print('Loaded data!')

    def convert_tokens_to_ids(self, tokens, vocab):
        ids = []
        for token in tokens:
            if token not in VOCAB_UPPER_CASE:
                token = token.lower()
            if token not in vocab:
                ids.append(vocab['<unkm>'])
            else:
                ids.append(vocab[token])
        return ids
    
    def pad_sequence(self, sequence, max_len, vocab):
        if len(sequence) < max_len:
            sequence = sequence + [vocab['<pad>']] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        assert len(sequence) == max_len
        return sequence
    def pad_list(self, sequence_list, max_len1, max_len2, vocab):
        if len(sequence_list) < max_len1:
            for i in range(max_len1 - len(sequence_list)):
                sequence_list += [[vocab['<pad>']] * max_len2]
        else:
            sequence_list = sequence_list[:max_len1]
        assert len(sequence_list) == max_len1
        return sequence_list

    def process_data(self, raw_diffs, raw_diff_atts, raw_diff_marks, raw_msgs, var_maps, raw_changes, raw_asts, raw_edge_change_codes, raw_edge_change_asts, raw_edge_ast_codes, raw_edge_asts):
        data_num = len(raw_diffs)
        diffs = []
        msgs = []
        msg_tars = []
        diff_atts = []
        diff_marks = []
        ast_changes = []
        edges = []
        sub_tokens = []
        max_diff_len = 0
        max_msg_len = 0
        max_att_len = 0
        max_ast_change_len = 0
        max_sub_token_len = 0        
        for i in tqdm(range(data_num)):
            raw_diff = raw_diffs[i]
            raw_diff_att = raw_diff_atts[i]
            raw_diff_mark = raw_diff_marks[i]
            raw_msg = raw_msgs[i]
            raw_change = raw_changes[i]
            raw_ast = raw_asts[i]
            raw_edge_change_code = raw_edge_change_codes[i]
            raw_edge_change_ast = raw_edge_change_asts[i]
            raw_edge_ast_code = raw_edge_ast_codes[i]
            raw_edge_ast = raw_edge_asts[i]

            var_map = var_maps[i]

            for j in range(len(raw_diff)):
                if raw_diff[j] in var_map:
                    raw_diff[j] = var_map[raw_diff[j]]
                if raw_diff[j] not in VOCAB_UPPER_CASE:
                    raw_diff[j] = raw_diff[j].lower()

            for j in range(len(raw_msg)):
                if raw_msg[j] in var_map:
                    raw_msg[j] = var_map[raw_msg[j]]
                if raw_msg[j] not in VOCAB_UPPER_CASE:
                    raw_msg[j] = raw_msg[j].lower()
                if raw_msg[j] in lemmatization:
                    raw_msg[j] = lemmatization[raw_msg[j]]

            max_diff_len = max(max_diff_len, len(raw_diff))
            diff = self.convert_tokens_to_ids(raw_diff, self.vocab)
            diff = [self.vocab['<start>']] + diff + [self.vocab['<eos>']]

            max_msg_len = max(max_msg_len, len(raw_msg))
            msg = self.convert_tokens_to_ids(raw_msg, self.vocab)
            msg_tar = msg
            msg =  [self.vocab['<start>']] + msg + [self.vocab['<eos>']]
            
            for j in range(len(raw_diff_att)):
                for k in range(len(raw_diff_att[j])):
                    assert raw_diff_att[j][k].islower()

            diff_att = []
            for j in range(len(raw_diff_att)):
                max_att_len = max(max_att_len, len(raw_diff_att[j]))
                diff_att.append(self.convert_tokens_to_ids(raw_diff_att[j], self.vocab))
            diff_att= [[]] + diff_att + [[]]

            diff_mark = [2] + raw_diff_mark + [2]
            assert len(diff) == len(diff_att) == len(diff_mark)

            diff = self.pad_sequence(diff, self.diff_len, self.vocab)
            msg = self.pad_sequence(msg, self.msg_len, self.vocab)
            for j in range(len(diff_att)):
                diff_att[j] = self.pad_sequence(diff_att[j], self.att_len, self.vocab)
            diff_att = self.pad_list(diff_att, self.diff_len, self.att_len, self.vocab)
            diff_mark = self.pad_sequence(diff_mark, self.diff_len, {'<pad>':0})

            raw_ast_change = raw_ast + raw_change
            max_ast_change_len = max(max_ast_change_len, len(raw_ast_change))
            ast_change = self.convert_tokens_to_ids(raw_ast_change, self.ast_change_vocab)
            ast_change = self.pad_sequence(ast_change, self.ast_change_len, self.ast_change_vocab)

            raw_edge_sub_token = []
            raw_sub_token = []
            map_sub_token = {}
            for j in range(len(raw_diff_att)):
                if raw_diff_att[j] == []:
                    continue
                cur_token = raw_diff[j]
                cur_att = raw_diff_att[j]
                start_sub_token = len(raw_sub_token)
                if cur_token in map_sub_token:
                    already = [raw_sub_token[k] for k in map_sub_token[cur_token]]
                    assert already == cur_att
                    for k in map_sub_token[cur_token]:
                        raw_edge_sub_token.append((j, k))
                else:
                    map_sub_token[cur_token] = []
                    raw_sub_token += cur_att
                    for k in range(len(cur_att)):
                        raw_edge_sub_token.append((j, start_sub_token + k))
                        map_sub_token[cur_token].append(start_sub_token + k)

            max_sub_token_len = max(max_sub_token_len, len(raw_sub_token))
            sub_token = self.convert_tokens_to_ids(raw_sub_token, self.vocab)
            sub_token = self.pad_sequence(sub_token, self.sub_token_len, self.vocab)
            
           
            for k in range(len(raw_msg)):
                msg_token = raw_msg[k]
                if msg_token in raw_diff:
                    msg_tar[k] = raw_diff.index(msg_token) + len(self.vocab) + 1
            

            
            for k in range(len(raw_msg)):
                msg_token = raw_msg[k]
                if msg_token in raw_sub_token:
                    loc = raw_sub_token.index(msg_token)
                    if msg_tar[k] >= len(self.vocab):
                        continue
                
                    msg_tar[k] = loc + len(self.vocab) + self.diff_len

            
            msg_tar =  [self.vocab['<start>']] + msg_tar + [self.vocab['<eos>']]
            msg_tar = self.pad_sequence(msg_tar, self.msg_len, self.vocab)            


            row = []
            col = []
            value = []
            ed = []
            # edge between code node and change edition  
            for edge in raw_edge_change_code:
                p1 = edge[0] + self.diff_len + self.sub_token_len + len(raw_ast)
                p2 = edge[1] + 1
                if p2 >= self.diff_len:
                    continue
                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 1)
            
            # edge between ast node and change edition
            for edge in raw_edge_change_ast:
                p1 = edge[0] + self.diff_len + self.sub_token_len + len(raw_ast)
                p2 = edge[1] + self.diff_len + self.sub_token_len

                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 2)
            
            # edge between ast node and code node
            for edge in raw_edge_ast_code:
                p1 = edge[0] + self.diff_len + self.sub_token_len
                p2 = edge[1] + 1
                if p2 >= self.diff_len:
                        continue
                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 3)

            # edge between ast nodes
            for edge in raw_edge_ast:
                p1 = edge[0] + self.diff_len + self.sub_token_len
                p2 = edge[1] + self.diff_len + self.sub_token_len

                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 4)

            # edge between code and sub token
            for edge in raw_edge_sub_token:
                p1 = edge[0] + 1
                p2 = edge[1] + self.diff_len

                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 6)


            # sequential 
            for j in range(len(raw_diff) + 2 - 1):
                p1 = j
                p2 = j + 1
                row, col, value, ed = process_edge(p1, p2, row, col, value, ed, 5)

           

            # self connection
            for i in range(self.graph_len):
                row.append(i)
                col.append(i)
                value.append(1)
                assert (i, i) not in ed

            deg_row = {}
            deg_col = {}
            for each_row in row:
                if each_row not in deg_row:
                    deg_row[each_row] = 1
                else:
                    deg_row[each_row] += 1
            for each_col in col:
                if each_col not in deg_col:
                    deg_col[each_col] = 1
                else:
                    deg_col[each_col] += 1

            for i in range(len(value)):
                value[i] = 1 / math.sqrt(deg_row[row[i]]) / math.sqrt(deg_col[col[i]])
            
            
            edge = sparse.coo_matrix((value, (row, col)), shape=(self.graph_len, self.graph_len))
            
            diffs.append(diff)
            msgs.append(msg)
            msg_tars.append(msg_tar)
            diff_atts.append(diff_att)
            diff_marks.append(diff_mark)
            ast_changes.append(ast_change)
            edges.append(edge)
            sub_tokens.append(sub_token)
        print(max_diff_len, max_msg_len, max_att_len, max_ast_change_len, max_sub_token_len)
        batches = [np.array(diffs), np.array(msgs), np.array(diff_atts), np.array(diff_marks), np.array(ast_changes), edges, np.array(msg_tars), np.array(sub_tokens)]
        index = list(range(num_train + num_valid + num_test))
        random.shuffle(index)
        train_index = index[:num_train]
        valid_index = index[num_train:num_train + num_valid]
        test_index = index[num_train + num_valid:]

        all_index = {'train':train_index, 'valid': valid_index, 'test': test_index}
        json.dump(all_index, open('all_index','w'))
        
        train_batches = []
        valid_batches = []
        test_batches = []

        for i in range(len(batches)):
            if i == 5:
                train_edges = [batches[i][x] for x in train_index]
                valid_edges = [batches[i][x] for x in valid_index]
                test_edges = [batches[i][x] for x in test_index]
                train_batches.append(train_edges)
                valid_batches.append(valid_edges)
                test_batches.append(test_edges)
            else:
                train_batches.append(batches[i][train_index])
                valid_batches.append(batches[i][valid_index])
                test_batches.append(batches[i][test_index])

        pickle.dump(train_batches, open("processed_train.pkl", 'wb'))
        pickle.dump(valid_batches, open("processed_valid.pkl", 'wb'))
        pickle.dump(test_batches, open("processed_test.pkl", 'wb'))

    def __getitem__(self, offset):
        data = []
        for i in range(len(self.data)):
            if i == 5:
                data.append(self.data[i][offset].toarray())
            else:
                data.append(self.data[i][offset])
        return data
    def __len__(self):
        return len(self.data[0])
def process_edge(p1, p2, row, col, value, ed, kind):
    if (p1, p2) not in ed:
        row.append(p1)
        col.append(p2)
        value.append(1)
        ed.append((p1, p2))
    if (p2, p1) not in ed:
        row.append(p2)
        col.append(p1)
        value.append(1)
        ed.append((p2, p1))
    return row, col, value, ed
