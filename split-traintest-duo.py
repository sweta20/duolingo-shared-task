import argparse

from utils import read_trans_prompts, read_transfile, FIELDSEP
from sklearn.model_selection import train_test_split
import pickle
import random

def get_data(fname: str, srclang: str, tgtlang: str, outputdir: str, extractref: bool, oversample: bool, topp: float, topk:int, all_train: bool) -> None:
    """
    This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)
    For training data, it combines the prompt with all accepted translations. 
    For dev or test data, it combines the prompt only with the most popular translation.
    """

    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    d = read_transfile(lines, strip_punc=False, weighted=True)
    id_text = dict(read_trans_prompts(lines))

    if all_train:
        train = list(d.keys())
        dev_size = 300.0/float(len(train))
        train, dev = train_test_split( train, test_size=dev_size, random_state=42)
        folds = ["train", "dev"]
    else:
        train, test = train_test_split( list(d.keys()), test_size=0.2, random_state=42)
        train, dev = train_test_split( train, test_size=0.1, random_state=42)
        folds = ["train", "dev", "test"]

    for fold in folds:
        data = eval(fold)
        srcfname = outputdir + "/" + fold + "." + srclang
        tgtfname = outputdir + "/" + fold + "." + tgtlang

        idfile = open(outputdir + "/" + fold + "-prompts", "w")

        with open(srcfname, "w") as src, open(tgtfname, "w") as tgt:
            for idstring in data:

                # prompt is combination of id and text.
                prompt = id_text[idstring]
                ats = d[idstring]

                # make sure that the first element is the largest.
                ats = sorted(ats.items(), key=lambda p: p[1], reverse=True)
                value = 0.0
                count = 0
                # if it is train
                if fold == "train":
                    # write all pairs.
                    for p in ats:
                        if not oversample:
                            print(prompt, file=src)
                            print(p[0], file=tgt)
                        else:
                            num = int(p[1]*1000)
                            for q in range(num):
                                print(prompt, file=src)
                                print(p[0], file=tgt)

                        value+=p[1]
                        count+=1
                        if value >= topp or count >= topk:
                            break

                else:
                    # write just the first pair (evaluate only on first line.)
                    top_ranked_text = ats[0][0]
                    print(idstring, file=idfile)
                    print(prompt, file=src)
                    print(top_ranked_text, file=tgt)

    if extractref:
        for fold in folds:
            outfile = outputdir + "/" + fold + "-all-accepted." + srclang + "-" + tgtlang + ".txt"
            with open(outfile, "w") as out:
                for idstring in eval(fold):
                    out.write(f"\n{idstring}{FIELDSEP}{id_text[idstring]}\n")
                    ats = d[idstring]
                    ats = sorted(ats.items(), key=lambda p: p[1], reverse=True)
                    for trans, weight in ats:
                        out.write(trans + FIELDSEP + str(weight) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)")
    parser.add_argument("--fname", help="Path of shared task file (probably something like train.en_vi.2020-01-13.gold.txt)", required=True)
    parser.add_argument("--srclang", help="Name of desired src language, probably something like en", required=True)
    parser.add_argument("--tgtlang", help="Name of desired tgt language, probably something like vi", required=True)
    parser.add_argument("--outputdir", help="Directory to store files", required=True)
    parser.add_argument("--extractref", help="Extract reference candidates in scoring format", action='store_true')
    parser.add_argument("--oversample", help="Oversample candidates based on LRF", action='store_true')
    parser.add_argument("--topp", help="Use only top p candidates", type=float, default=1.0)
    parser.add_argument("--topk", help="Use only top k candidates", type=int, default=10000)
    parser.add_argument("--all", help="Use all as training data", action='store_true')
    args = parser.parse_args()

    get_data(args.fname, args.srclang, args.tgtlang, args.outputdir, args.extractref, args.oversample, args.topp, args.topk, args.all)
