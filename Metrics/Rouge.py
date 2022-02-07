import argparse
import json
import os

### Rouge
def get_rouge(ref_path, gen_path, is_sentence=False):
    if is_sentence:
        evaluate_cmd = "sumeval r-nl \"{}\" \"{}\"".format(gen_path, ref_path)
    else:
        evaluate_cmd = "sumeval r-nl -f \"{}\" \"{}\" -in".format(gen_path, ref_path)
    rouge_score = json.load(os.popen(evaluate_cmd))["averages"]
    for key in rouge_score.keys():
        rouge_score[key] = 100*rouge_score[key]
    return rouge_score

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate Rouge-1, Rouge-2, Rouge-N by sumeval')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        print(get_rouge(args.ref_path, args.gen_path))
    else:
        print("File not exits")