import json
import sys
import numpy as np
from tqdm import tqdm
import pickle
import traceback
from copy import deepcopy
import signal
import time
import os
import javalang
from get_ast_root_action import Node, get_ast_root, ActionNode, get_ast_action
MODIFIERS = ['abstract', 'default', 'final', 'native', 'private',
                  'protected', 'public', 'static', 'strictfp', 
                  'transient', 'volatile'] 
WASTE_TIME = json.load(open('WASTE_TIME'))
CHANGE_SINGLE = json.load(open('CHANGE_SINGLE'))


def process_bracket(tokens):
    if tokens[0] == '}':
        tokens.pop(0)
    stack = []
    for token in tokens:
        if token == '{':
            stack.append('{')
        elif token == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                stack.append('}')
    l_num = stack.count('{')
    r_num = stack.count('}')
    tokens = ['{'] * r_num + tokens + ['}'] * l_num
    return tokens

def get_ast(code, file_name):
    if code in CHANGE_SINGLE[0]:
        code = CHANGE_SINGLE[1][CHANGE_SINGLE[0].index(code)]
    text = ' '.join(code)
    text = text.replace('COMMENT', ' ')
    text = text.replace('SINGLE', ' ')
    text = text.replace('<nl>', ' ')
    text = text.replace('<nb>', ' ')
    if len(text) == 0:
        return None, -1
    try:
        tokens_ori = list(javalang.tokenizer.tokenize(text))
    except:
        return None, -1
    codes_ori = [x.value for x in tokens_ori]
    
    if len(codes_ori) == 0:  
        return None, -1
    
    if 'implement' in codes_ori:
        codes_ori.remove('implement')
    if codes_ori[-1] == 'implements':
        codes_ori.remove('implements')
    if len(codes_ori) == 0:
        return None, -1
    
    
    if len(codes_ori) >= 4 and 'class' in codes_ori and codes_ori[-2] == '<' and codes_ori[-1] != '>':
        codes_ori += '>' 
    
    codes_ori = process_bracket(codes_ori)
    
    if len(codes_ori) == 0:
        return None, -1
    
    ori_start_token = ' '.join(codes_ori)
    
    if codes_ori[0] == 'import':
        pass
    elif codes_ori[0] == 'package':
        pass
    elif codes_ori[0] == '@':
        if 'class' in codes_ori:  # definition of class
            pass
        else:  # definition of method
            codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
            # gumtree can only parse class, so a padding class needs to be inserted
    elif codes_ori[0] in MODIFIERS:
        
        if 'class' in codes_ori:  # definition of class
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                raise
            else:
                codes_ori +=  ['{', '}'] 
        elif '(' in codes_ori and ')' in codes_ori and ('=' not in codes_ori or ('='  in codes_ori and codes_ori.index('(') < codes_ori.index('=') and codes_ori.index(')') < codes_ori.index('='))):  # definition of method
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                raise
            elif codes_ori[-1] != ';':
                codes_ori +=  ['{', '}'] 
                
            codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
        else:  # definition of field
            codes_ori = ['class', 'pad_pad_class', '{', '{'] + codes_ori + ['}', '}']
    elif codes_ori[0] == '{':
        codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
    else:
        if codes_ori[0] == 'if':
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                
                raise
            elif codes_ori[-1] == ')':
                codes_ori +=  ['{', '}']
        codes_ori = ['class', 'pad_pad_class', '{', '{'] + codes_ori + ['}', '}']

  
    text = ' '.join(codes_ori)
    start_code_pos = text.index(ori_start_token)
    assert start_code_pos != -1

    tokens = list(javalang.tokenizer.tokenize(text))
    if [x.value for x in tokens] in WASTE_TIME:
        return None, -1

    open(tem_path + '/%s.java'%file_name, 'w').write(text)
    
    root = get_ast_root(file_name, tem_path)

    return root, start_code_pos

def get_edge_ast_code(root, codes, start_code_pos): 
    edge_ast_code = []
    edge_ast = []
    dmap_ast = {}  
    nodes = root.get_all_nodes()
    ast_tokens = []
    
    start_index = {}
    pos_index = {}
    dmap_node_code = {}
    for node in nodes:
        if node.pos < start_code_pos:
            continue
        if node.pos == start_code_pos and (node.typeLabel == "CompilationUnit" or node.typeLabel == "Block"):
            continue
        if len(node.children) == 0 and node.typeLabel != 'Block' :

            name = node.label
            if name not in start_index:
                start_cur = -1
            else: 
                start_cur = start_index[name]
                
                if pos_index[name] >= node.pos:
                    continue
            if name not in codes:
                continue
            if m == 70 and name == 'nextParent' and start_cur == -1:
                code_no = codes.index('nextParent:', start_cur + 1)
            else:
                try:
                    code_no = codes.index(name, start_cur + 1)
                except:
                    continue
            dmap_node_code[node.ori_id] = code_no

            start_index[name] = code_no
            pos_index[name] = node.pos
            
            ast_no = dmap_ast[node.father.ori_id]
            edge_ast_code.append((ast_no, code_no))
        else:
            dmap_ast[node.ori_id] = len(ast_tokens)
            ast_tokens.append(node.typeLabel)
            if node.father.pos < start_code_pos:
                continue
            if node.father.pos == start_code_pos and (node.father.typeLabel == "CompilationUnit" or node.father.typeLabel == "Block"):
                continue
            edge_ast.append((dmap_ast[node.father.ori_id], dmap_ast[node.ori_id]))
    verify_only = []
    for node in dmap_node_code:
        assert dmap_node_code[node] not in verify_only
        verify_only.append(dmap_node_code[node])
    return edge_ast_code, edge_ast, ast_tokens, dmap_ast, dmap_node_code
            
def get_edge_update(codes_old, codes_new):
    edge_ast_code_old = []
    edge_ast_old = []
    ast_tokens_old = []
    edge_ast_code_new = []
    edge_ast_new = []
    ast_tokens_new = []
    edge_change_code_old = []
    edge_change_ast_old = []
    edge_change_code_new = []
    edge_change_ast_new = []
    change = []

    global time_stamp
    root_old, start_code_pos_old = get_ast(codes_old, 'old_%d'%time_stamp)
    if root_old == None:
        pass
    else:    
        edge_ast_code_old, edge_ast_old, ast_tokens_old, dmap_ast_old, dmap_node_code_old = get_edge_ast_code(root_old, codes_old, start_code_pos_old)
    
    root_new, start_code_pos_new = get_ast(codes_new, 'new_%d'%time_stamp)
    if root_new == None:
        pass
    else:
        edge_ast_code_new, edge_ast_new, ast_tokens_new, dmap_ast_new, dmap_node_code_new = get_edge_ast_code(root_new, codes_new, start_code_pos_new)
    
    if root_old == None or root_new == None:
        return edge_ast_code_old, edge_ast_old, ast_tokens_old, \
                edge_ast_code_new, edge_ast_new, ast_tokens_new, \
                edge_change_code_old, edge_change_ast_old, \
                edge_change_code_new, edge_change_ast_new, change
    
    all_match_new, all_delete, all_add = get_ast_action('old_%d'%time_stamp, 'new_%d'%time_stamp, root_old, root_new, tem_path)

    

    all_nodes1 = root_old.get_all_nodes()
    all_nodes2 = root_new.get_all_nodes()
    map1 = {}
    map2 = {}
    for i, node in enumerate(all_nodes1):
        map1[node.ori_id] = node.idx 
        assert i == node.idx
    for i, node in enumerate(all_nodes2):
        map2[node.ori_id] = node.idx
        assert i == node.idx
    change = []
    edge_change_code_old = []
    edge_change_code_new = []
    edge_change_ast_old = []
    edge_change_ast_new = []
    for cur_match in all_match_new:
        start_change = len(change)

        kind = cur_match[0]
        old_node = cur_match[1]
        new_node = cur_match[2]
        old_idx = old_node.idx
        assert all_nodes1[map1[old_idx]].label == old_node.name
        assert all_nodes1[map1[old_idx]].typeLabel == old_node.typ
        new_idx = new_node.idx
        assert all_nodes2[map2[new_idx]].label == new_node.name
        assert all_nodes2[map2[new_idx]].typeLabel == new_node.typ
        if old_idx in dmap_node_code_old:
            if new_idx not in dmap_node_code_new:
                continue
            edge_change_code_old.append((start_change, dmap_node_code_old[old_idx]))   
            edge_change_code_new.append((start_change, dmap_node_code_new[new_idx]))
            change.append(kind)
        elif old_idx in dmap_ast_old:
            if new_idx not in dmap_ast_new:
                continue
            
            edge_change_ast_old.append((start_change, dmap_ast_old[old_idx]))
            edge_change_ast_new.append((start_change, dmap_ast_new[new_idx]))
            change.append(kind)
    for old_node in all_delete:
        start_change = len(change)

        old_idx = old_node.idx
        assert all_nodes1[map1[old_idx]].label == old_node.name
        assert all_nodes1[map1[old_idx]].typeLabel == old_node.typ
        if old_idx in dmap_node_code_old:
            edge_change_code_old.append((start_change, dmap_node_code_old[old_idx]))
            change.append('delete')
        elif old_idx in dmap_ast_old:
            edge_change_ast_old.append((start_change, dmap_ast_old[old_idx]))
            change.append('delete')
    for cur_add in all_add:
        start_change = len(change)

        new_node = cur_add[0]
        new_idx = new_node.idx
        assert all_nodes2[map2[new_idx]].label == new_node.name
        assert all_nodes2[map2[new_idx]].typeLabel == new_node.typ
        if new_idx in dmap_node_code_new:
            edge_change_code_new.append((start_change, dmap_node_code_new[new_idx]))
            change.append('add')
        elif new_idx in dmap_ast_new:
            edge_change_ast_new.append((start_change, dmap_ast_new[new_idx]))
            change.append('add')
    if root_old:
        os.system('rm %s/old_%d.java'%(tem_path, time_stamp))
        os.system('rm %s/old_%d.ast'%(tem_path, time_stamp))
    if root_new:
        os.system('rm %s/new_%d.java'%(tem_path, time_stamp))
        os.system('rm %s/new_%d.ast'%(tem_path, time_stamp))
    time_stamp += 1
    return edge_ast_code_old, edge_ast_old, ast_tokens_old, \
                edge_ast_code_new, edge_ast_new, ast_tokens_new, \
                edge_change_code_old, edge_change_ast_old, \
                edge_change_code_new, edge_change_ast_new, change

def get_edge_normal(codes_old):
    edge_ast_code_old = []
    edge_ast_old = []
    ast_tokens_old = []
    global time_stamp
    root_old, start_code_pos_old = get_ast(codes_old, 'old_%d'%time_stamp)
    if root_old == None:
        pass
    else:
        edge_ast_code_old, edge_ast_old, ast_tokens_old, dmap_ast_old, dmap_node_code_old = get_edge_ast_code(root_old, codes_old, start_code_pos_old)

    if root_old:
        os.system('rm %s/old_%d.java'%(tem_path, time_stamp))
        os.system('rm %s/old_%d.ast'%(tem_path, time_stamp))
    time_stamp += 1
    
    return edge_ast_code_old, edge_ast_old, ast_tokens_old


if __name__ == '__main__':

    
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    postfix = '_%d_%d'%(begin, end)


    tem_path = 'TEMPORARY_FILES/TEMPORARY_FILES%s'%postfix
    if not os.path.exists(tem_path):
        os.makedirs(tem_path)
    try:
        time_stamp = 0 

        difftokens = json.load(open('../DataSet/difftoken.json'))
        processed_tokens = pickle.load(open('processed_tokens.pkl', 'rb'))
        processed_types = pickle.load(open('processed_types.pkl', 'rb'))

        total_change = []
        total_ast = []
        total_edge_change_code = []  
        total_edge_change_ast = []  
        total_edge_ast_code = []  
        total_edge_ast = []  
        
        for m, lines_tokens in enumerate(tqdm(processed_tokens[begin:end])):
            m += begin
            all_change = []
            all_ast = []
            all_edge_change_code = []  
            all_edge_change_ast = []  
            all_edge_ast_code = []  
            all_edge_ast = []  
            
            all_token = []


            types = processed_types[m]

            for k, tokens in enumerate(lines_tokens):
                code_old = []
                code_new = []
                ast_old = []
                ast_new = []
                change = []
                edge_change_code = []  
                edge_change_ast = []  
                edge_ast_code = []  
                edge_ast = []  

                all_start_code = len(all_token)
                all_start_change = len(all_change)
                all_start_ast = len(all_ast)
                if types[k] == 100:  
                    edge_ast_code_old, edge_ast_old, ast_old, \
                    edge_ast_code_new, edge_ast_new, ast_new, \
                    edge_change_code_old, edge_change_ast_old, \
                    edge_change_code_new, edge_change_ast_new, change = get_edge_update(tokens[0], tokens[1])

                    for edge in edge_ast_code_old:
                        edge_ast_code.append((all_start_ast + edge[0], all_start_code + edge[1]))
                    for edge in edge_ast_old:
                        edge_ast.append((all_start_ast + edge[0], all_start_ast + edge[1]))
                    for edge in edge_change_code_old:
                        edge_change_code.append((all_start_change + edge[0], all_start_code + edge[1]))
                    for edge in edge_change_ast_old:
                        edge_change_ast.append((all_start_change + edge[0], all_start_ast + edge[1]))
                    for edge in edge_ast_code_new:
                        edge_ast_code.append((all_start_ast + len(ast_old) + edge[0], all_start_code + len(tokens[0]) + edge[1]))
                    for edge in edge_ast_new:
                        edge_ast.append((all_start_ast + len(ast_old) + edge[0], all_start_ast + len(ast_old) + edge[1]))
                    for edge in edge_change_code_new:
                        edge_change_code.append((all_start_change + edge[0], all_start_code + len(tokens[0]) + edge[1]))
                    for edge in edge_change_ast_new:
                        edge_change_ast.append((all_start_change + edge[0], all_start_ast + len(ast_old) + edge[1]))

                    code_old = tokens[0]
                    code_new = tokens[1]

                else:
                    assert types[k] == -1 or types[k] == 1 or types[k] == 0
                    edge_ast_code_old, edge_ast_old, ast_old = get_edge_normal(tokens)

                    for edge in edge_ast_code_old:
                        edge_ast_code.append((all_start_ast + edge[0], all_start_code + edge[1]))
                    for edge in edge_ast_old:
                        edge_ast.append((all_start_ast + edge[0], all_start_ast + edge[1]))

                    code_old = tokens
                    assert len(code_old) > 0
                
                all_token += code_old
                all_token += code_new
                all_ast += ast_old
                all_ast += ast_new
                all_change += change
                all_edge_change_code += edge_change_code  
                all_edge_change_ast += edge_change_ast
                all_edge_ast_code += edge_ast_code
                all_edge_ast += edge_ast

            assert all_token == difftokens[m]
            total_change.append(all_change)
            total_ast.append(all_ast)
            total_edge_change_code.append(all_edge_change_code)  
            total_edge_change_ast.append(all_edge_change_ast)
            total_edge_ast_code.append(all_edge_ast_code)
            total_edge_ast.append(all_edge_ast)
        
        output_path = 'DataSet_temporary/DataSet_%s'%postfix
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        json.dump(total_change, open(output_path + '/change.json', 'w'))
        json.dump(total_ast, open(output_path + '/ast.json', 'w'))
        json.dump(total_edge_change_code, open(output_path + '/edge_change_code.json', 'w'))
        json.dump(total_edge_change_ast, open(output_path + '/edge_change_ast.json', 'w'))
        json.dump(total_edge_ast_code, open(output_path + '/edge_ast_code.json', 'w'))
        json.dump(total_edge_ast, open(output_path + '/edge_ast.json', 'w'))

    except:
        if not os.path.exists('ERROR'):
            os.makedirs('ERROR')
        error = traceback.format_exc()
        open('ERROR/error%s'%postfix, 'w').write(error + '\n' + str(m))       