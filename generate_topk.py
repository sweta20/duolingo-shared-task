import argparse
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from staple_2020_scorer import score
from utils import read_transfile, read_file, read_trans_prompts, FIELDSEP
import string

table = str.maketrans(dict.fromkeys(string.punctuation))

SCORE_FIELD = 3
TRANS_FIELD = 1

def filter_pred(pred, t):
    filtered_list = {}
    for idstring in pred.keys():
        ats = sorted(pred[idstring].items(), key=lambda p: p[1], reverse=True)
        filtered_list[idstring] = {}
        for k, v in ats[:t]:
            filtered_list[idstring][k] = v
    return filtered_list

def write_to_file(d, id_text, filename):
    with open(filename, "w") as out:
        for prompt in d:
            src = id_text[prompt]
            out.write(f"\n{prompt}{FIELDSEP}{src}\n")
            for para, v in d[prompt].items():
                out.write(para + "\n")

def generate_topk(args):
    
    with open(args.predfile) as f:
        print("reading predfile")
        data = read_transfile(f.readlines(), weighted=False)

    with open(args.predfile) as f:
        print("reading predfile")
        id_text = dict(read_trans_prompts(f.readlines())) 

    if args.goldfile is not None:
        with open(args.goldfile) as f:
            print("reading goldfile")
            gold = read_transfile(f.readlines(), weighted=True)

    features = read_file(args.featfile)
    score_dict = {}
    for line in features:
        fields = [f.strip() for f in line.split(' ||| ')]
        # print(fields)
        trans = fields[TRANS_FIELD]
        trans = trans.translate(table)
        score_dict[trans] = float(fields[SCORE_FIELD])

    new_pred = {}
    for prompt in data:
        new_pred[prompt] = {}
        for para, v in data[prompt].items():
            new_pred[prompt][para] = score_dict[para]


    if args.k==-1:
        scores_F1= {}
        for k in range(2, len(data[prompt].items()), 1):
            new_pred_fil = filter_pred(new_pred, k)
            macro_weighted_f1 = score(gold, new_pred_fil)
            scores_F1[k] = macro_weighted_f1
        best_k =  max(scores_F1, key=scores_F1.get)

        with open(args.thresholdfile, "w") as f:
            f.write(str(best_k))

    else:
        new_pred_fil = filter_pred(new_pred, args.k)
        # macro_weighted_f1 = score(gold, new_pred_fil)
        write_to_file(new_pred_fil, id_text, args.outputfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This selects threshold based on best scoring k value on dev")
    parser.add_argument("--featfile", help="Path to features file", required=True)
    parser.add_argument("--predfile", help="File to write output to", required=True)
    parser.add_argument("--goldfile", help="Gold annotations", required=False, default=None)
    parser.add_argument("--thresholdfile", help="File to optimal threshold to", required=False, default=None)
    parser.add_argument("--outputfile", help="File to write output to", required=False, default=None)
    parser.add_argument("--k", help="File to write output to",type=int, required=False, default=-1)
    # parser.add_argument("--score-k", action='store_true')
    args = parser.parse_args()

    generate_topk(args)

