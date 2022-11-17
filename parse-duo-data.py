import argparse

from utils import read_trans_prompts, read_transfile, FIELDSEP
from sklearn.model_selection import train_test_split
import pickle

def main(srcfile: str, outdir: str):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 
    """

    with open(srcfile, encoding="utf-8") as f:
        lines = f.readlines()
    d = read_transfile(lines, strip_punc=False, weighted=True)
    id_text = dict(read_trans_prompts(lines))

    with open(outdir + "/gold-src", "w") as fsrc, open(outdir + "/gold-tgt", "w") as ftgt, open(outdir + "/gold-prompts", "w") as fprompts,  open(outdir + "/gold-scores", "w") as fscore:
        for prompt in d:
            src_out = id_text[prompt]
            for para, score in d[prompt]:
                fsrc.write(f"{src_out}\n")
                fprompts.write(f"{prompt}\n")
                ftgt.write(f"{para}\n")
                fscore.write(f"{score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This processes the output of sockeye-translate so that it can be scored with sacrebleu and so that it has the shared task format. ")
    parser.add_argument("--srcfile", help="Name of intput file to sockeye-translate", required=True)
    parser.add_argument("--outdir", help="Name of desired output file. This will be the shared task format file.", required=True)
    args = parser.parse_args()

    main(args.srcfile, args.outdir)
