import subprocess
import pickle
import math, json
import time
import os
all_num = len(pickle.load(open('processed_tokens.pkl', 'rb')))
each_num = 100
total_change = []
total_ast = []
total_edge_change_code = []  
total_edge_change_ast = []  
total_edge_ast_code = []  
total_edge_ast = []  
for i in range(math.ceil(all_num / each_num)):
    begin = i * each_num
    end = min((i + 1) * each_num, all_num)
    postfix = '_%d_%d'%(begin, end)
    output_path = 'DataSet_temporary/DataSet_%s'%postfix
    cur_change = json.load(open(output_path + '/change.json'))
    cur_ast = json.load(open(output_path + '/ast.json'))
    cur_edge_change_code = json.load(open(output_path + '/edge_change_code.json'))
    cur_edge_change_ast = json.load(open(output_path + '/edge_change_ast.json'))
    cur_edge_ast_code = json.load(open(output_path + '/edge_ast_code.json'))
    cur_edge_ast = json.load(open(output_path + '/edge_ast.json'))

    total_change += cur_change
    total_ast += cur_ast
    total_edge_change_code += cur_edge_change_code
    total_edge_change_ast += cur_edge_change_ast
    total_edge_ast_code += cur_edge_ast_code
    total_edge_ast += cur_edge_ast 

assert len(total_change) == len(total_ast) == len(total_edge_change_code) == len(total_edge_change_ast) == len(total_edge_ast_code) == len(total_edge_ast) == all_num

dataset_path = 'DataSet'
if not os.path.exists('../' + dataset_path):
    os.makedirs('../' + dataset_path)
json.dump(total_change, open('../%s/change.json'%dataset_path, 'w'))
json.dump(total_ast, open('../%s/ast.json'%dataset_path, 'w'))
json.dump(total_edge_change_code, open('../%s/edge_change_code.json'%dataset_path, 'w'))
json.dump(total_edge_change_ast, open('../%s/edge_change_ast.json'%dataset_path, 'w'))
json.dump(total_edge_ast_code, open('../%s/edge_ast_code.json'%dataset_path, 'w'))
json.dump(total_edge_ast, open('../%s/edge_ast.json'%dataset_path, 'w'))