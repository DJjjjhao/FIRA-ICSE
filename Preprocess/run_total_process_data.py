import subprocess
import pickle
import math
import json
import time
import os
from tqdm import tqdm 
def process():

    difftokens = json.load(open('../DataSet/difftoken.json'))
    diffmarks = json.load(open('../DataSet/diffmark.json'))
    processed_tokens = []
    processed_types = []
    for i, tokens in enumerate(tqdm(difftokens)):
        
        marks = diffmarks[i]
        state = '<start>'
        
        processed_token = []
        processed_type = []
        delete_token = []
        add_token = []
        normal_token = []
        j = 0
        while j < len(tokens):
            token = tokens[j]
            mark = marks[j]
            
            if token == '<nb>':
                if state == 0:
                    assert normal_token
                    processed_token.append(normal_token)
                    processed_type.append(0)
                elif state == -1:
                    assert delete_token
                    processed_token.append(delete_token)
                    processed_type.append(-1)
                elif state == 1:
                    assert add_token
                    if delete_token == []:
                        processed_token.append(add_token)
                        processed_type.append(1)
                    else:
                        processed_token.append([delete_token, add_token])
                        processed_type.append(100)
                assert mark == 2
                
                end_nb = tokens[j:].index('<nl>') + j
                for jj in range(j, end_nb + 1):
                    assert marks[jj] == 2
                processed_token.append(tokens[j:end_nb + 1])
                processed_type.append(0)
                
                state = '<start>'
                delete_token = []
                add_token = []
                normal_token = []
                j = end_nb + 1
                continue

            if state == '<start>':
                if mark == 1:
                    delete_token.append(token)
                    state = -1
                elif mark == 3:
                    add_token.append(token)
                    state = 1
                elif mark == 2:
                    normal_token.append(token)
                    state = 0
            elif state == 0:
                if mark == 1:
                    processed_token.append(normal_token)
                    processed_type.append(0)
                    normal_token = []
                    delete_token.append(token)
                    state = -1
                elif mark == 3:
                    processed_token.append(normal_token)
                    processed_type.append(0)
                    normal_token = []
                    add_token.append(token)
                    state = 1
                elif mark == 2:
                    normal_token.append(token)
                    state = 0
            elif state == -1:
                if mark == 1:
                    delete_token.append(token)
                    state = -1
                elif mark == 3:
                    add_token.append(token)
                    state = 1
                elif mark == 2:
                    processed_token.append(delete_token)
                    processed_type.append(-1)
                    delete_token = []
                    normal_token.append(token)
                    state = 0
            elif state == 1:
                if mark == 1:
                    if delete_token == []:
                        processed_token.append(add_token)
                        processed_type.append(1)
                        add_token = []
                        delete_token.append(token)
                        state = -1
                    else:
                        processed_token.append([delete_token, add_token])
                        processed_type.append(100)

                        delete_token = []
                        add_token = []
                        delete_token.append(token)
                        state = -1
                elif mark == 3:
                    add_token.append(token)
                    state = 1
                elif mark == 2:
                    if delete_token == []:
                        processed_token.append(add_token)
                        processed_type.append(1)
                        add_token = []
                        normal_token.append(token)
                        state = 0
                    else:
                        processed_token.append([delete_token, add_token])
                        processed_type.append(100)

                        delete_token = []
                        add_token = []
                        normal_token.append(token)
                        state = 0

            j += 1
        if state == 0:
            assert normal_token
            processed_token.append(normal_token)
            processed_type.append(0)
        elif state == -1:
            assert delete_token
            processed_token.append(delete_token)
            processed_type.append(-1)
        elif state == 1:
            assert add_token
            if delete_token == []:
                processed_token.append(add_token)
                processed_type.append(1)
            else:
                processed_token.append([delete_token, add_token])
                processed_type.append(100)

        processed_tokens.append(processed_token)
        processed_types.append(processed_type)

   
    pickle.dump(processed_tokens, open('processed_tokens.pkl', 'wb'))
    pickle.dump(processed_types, open('processed_types.pkl', 'wb'))

if __name__ == '__main__':
    if not os.path.exists('processed_tokens.pkl'):
        process()
    all_num = len(pickle.load(open('processed_tokens.pkl', 'rb')))

    jobs = []
    max_num = 100
    each_num = 100

    for i in range(math.ceil(all_num / each_num)):
        while True:
            run_num = 0 
            for x in jobs:
                if x.poll() is None:
                    run_num += 1
            if run_num < max_num:
                break
            time.sleep(1)
        start = i * each_num
        end = min((i + 1) * each_num, all_num)
        p = subprocess.Popen("python process_data_ast_parallel.py %d %d"%(start, end), shell=True)
        jobs.append(p)
        time.sleep(1)
    for job in jobs:
        job.wait()




