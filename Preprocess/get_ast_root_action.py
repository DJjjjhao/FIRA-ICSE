import json
import pickle
import subprocess
from copy import deepcopy
all_keys = ['id', 'type', 'typeLabel', 'pos', 'length', 'children','label' ] 

class Node:
    def __init__(self):
        self.ori_id = None
        self.idx = None
        self.type = None
        self.label = None
        self.typeLabel = None
        self.pos = None
        self.length = None
        self.children = []
        self.father = None
    def __str__(self):
        return 'ori_id:%s idx:%s type:%s label:%s typeLabel:%s pos:%s length:%s'%(self.ori_id, self.idx, self.type, self.label, self.typeLabel, self.pos, self.length)
    def print_tree(self, depth):
        strr = '\t' * depth + str(self) + '\n'
        for child in self.children:
            strr += child.print_tree(depth + 1)
        return strr
    def get_all_nodes(self):
        nodes = []
        nodes.append(self)
        for node in self.children:
            nodes += node.get_all_nodes()
        return nodes

class ActionNode:
    def __init__(self, typ, idx, name=None):
        self.typ = typ
        self.idx = idx
        self.name = name
    
    def __eq__(self, node):
        return self.typ == node.typ and self.idx == node.idx and self.name == node.name

def process_ast(ast):
    nodes = []
    
    node = Node()
    if 'label' in ast:
        node.label = ast['label']
    else:
        node.label = None
    
    node.ori_id = int(ast['id'])
    node.type = ast['type']
    node.typeLabel = ast['typeLabel']
    node.pos = int(ast['pos'])
    node.length = ast['length']

    if node.typeLabel == 'NullLiteral':
        assert node.label == None
        node.label = 'null'
    if node.typeLabel == 'ThisExpression':
        assert node.label == None
        node.label = 'this'
    nodes.append(node)

    for child in ast["children"]:
        nodes += process_ast(child)
        nodes.append('^')
    return nodes

def get_ast_root(file_name, tem_path):
    out = subprocess.Popen('../gumtree/gumtree/bin/gumtree parse %s/%s.java'%(tem_path, file_name), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,_ = out.communicate()
    try:
        ast = json.loads(stdout.decode('utf-8'))
    except:
        return None
    json.dump(ast, open('%s/%s.ast'%(tem_path, file_name), 'w'), indent=1)
    root = Node()
    root.label = 'root'
    root.pos = -1
    all_nodes = []
    all_nodes.append(root)
    all_nodes += process_ast(ast['root'])
    all_nodes += ['^']

    all_nodes_new = []
    root = all_nodes[0]
    root.idx = 0
    all_nodes_new.append(root)
    cur_node = root
    idx = 1
    for node in all_nodes[1:]:
        if node == '^':
            cur_node = cur_node.father
        else:
            node.idx = idx
            node.father = cur_node
            all_nodes_new.append(node)
            cur_node.children.append(node)
            cur_node = node
            idx += 1
    return root

def get_typ_idx(strr):
    if ':' in strr:
        typ, name_idx = strr.split(':')
        typ = typ.strip()
        name_idx = name_idx.strip()
        name = name_idx[:name_idx.index('(')]
        idx = name_idx[name_idx.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        return ActionNode(typ, int(idx), name)

    else:    
        typ = strr[:strr.index('(')]
        idx = strr[strr.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        if typ == 'NullLiteral':
            return ActionNode(typ, int(idx), 'null')
        if typ == 'ThisExpression':
            return ActionNode(typ, int(idx), 'this')
        return ActionNode(typ, int(idx))

def get_ast_action(file_name1, file_name2, root1, root2, tem_path):
    out = subprocess.Popen('../gumtree/gumtree/bin/gumtree diff %s/%s.java %s/%s.java'%(tem_path, file_name1, tem_path, file_name2), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    stdout = stdout.decode('utf-8')
    raw_actions = [x.strip() for x in stdout.splitlines() if x.strip()]
    all_match = []
    all_delete = []
    all_update = []
    all_move = []
    all_add = []
    for i, raw_action in enumerate(raw_actions):
        if raw_action.startswith('Match'):
            raw_node_old, raw_node_new = raw_action.lstrip('Match').split(' to ')
            raw_node_old = raw_node_old.strip()
            raw_node_new = raw_node_new.strip()
            node_old = get_typ_idx(raw_node_old)
            node_new = get_typ_idx(raw_node_new)
            all_match.append((node_old, node_new))
        elif raw_action.startswith('Delete'):
            raw_node_old = raw_action.lstrip('Delete')
            raw_node_old = raw_node_old.strip()
            node_old = get_typ_idx(raw_node_old)
            all_delete.append(node_old)
        elif raw_action.startswith('Update'):
            raw_node_old, new_name = raw_action.lstrip('Update').split(' to ')
            raw_node_old = raw_node_old.strip()
            new_name = new_name.strip()
            node_old = get_typ_idx(raw_node_old)
            all_update.append((node_old, new_name))
        elif raw_action.startswith('Move'):
            raw_node_old, tem = raw_action.lstrip('Move').split(' into ')
            raw_node_new, new_pos = tem.split(' at ')
            raw_node_old = raw_node_old.strip()
            raw_node_new = raw_node_new.strip()
            new_pos = new_pos.strip()
            node_old = get_typ_idx(raw_node_old)
            node_new = get_typ_idx(raw_node_new)
            new_pos = int(new_pos)
            all_move.append((node_old, node_new, new_pos))
        elif raw_action.startswith('Insert'):
            raw_node_new, tem = raw_action.lstrip('Insert').split(' into ')
            raw_node_par, pos = tem.split(' at ')
            raw_node_new = raw_node_new.strip()
            raw_node_par = raw_node_par.strip()
            pos = pos.strip()
            node_new = get_typ_idx(raw_node_new)
            node_par = get_typ_idx(raw_node_par)
            pos = int(pos)
            all_add.append((node_new, node_par, pos))

    all_nodes1 = root1.get_all_nodes()
    all_nodes2 = root2.get_all_nodes()
    map1 = {}
    map2 = {}  
    for i, node in enumerate(all_nodes1):
        map1[node.ori_id] = i 
        assert i == node.idx
    
    for i, node in enumerate(all_nodes2):
        map2[node.ori_id] = i 
        assert i == node.idx
    
    document_move = [False] * len(all_move)
    document_update = [False] * len(all_update)
    all_match_new = []
    for i in range(len(all_match)):
        cur_match = all_match[i]
        move_flag = False
        update_flag = False
        
        for j in range(len(all_update)):
            cur_update = all_update[j]
            if cur_update[0] == cur_match[0]:
                value1 = cur_update[1]
                value2 = cur_match[1].name
                assert value1 == value2
                update_flag = True
                document_update[j] = True
                break
        for j in range(len(all_move)):
            cur_move = all_move[j]
            if cur_move[0] == cur_match[0]:
                parent_id = cur_move[1].idx
                child_id = cur_match[1].idx
                children_id = [x.ori_id for x in all_nodes2[map2[parent_id]].children]
                assert child_id in children_id
                move_flag = True
                document_move[j] = True
                if update_flag == False:
                    assert cur_match[0].typ == cur_match[1].typ
                    assert cur_match[0].name == cur_match[1].name
                break
        if move_flag == False and update_flag == False:
            all_match_new.append(('match', cur_match[0], cur_match[1]))
        elif move_flag == False and update_flag == True:
            all_match_new.append(('update', cur_match[0], cur_match[1]))
        elif move_flag == True and update_flag == False:
            all_match_new.append(('move', cur_match[0], cur_match[1]))
        elif move_flag == True and update_flag == True:
            all_match_new.append(('update', cur_match[0], cur_match[1]))
    
    assert sum(document_move) == len(all_move)
    assert sum(document_update) == len(all_update)
    for i in range(len(all_add)):
        cur_add = all_add[i]
        child_id = cur_add[0].idx
        parent_id = cur_add[1].idx
        children_id = [x.ori_id for x in all_nodes2[map2[parent_id]].children]
        assert child_id in children_id
    return all_match_new, all_delete, all_add

