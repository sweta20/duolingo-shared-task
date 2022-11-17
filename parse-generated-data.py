import argparse
import json
import sentencepiece as spm

FIELDSEP = "|"

def read_file(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def main(srcfile: str, tgtfile: str, outdir: str, promptsfile: str):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 
    """

    prompts = read_file(promptsfile)
    srcdata = read_file(srcfile)
    tgtdata = read_file(tgtfile)

    assert len(prompts) == len(srcdata) == len(tgtdata)

    with open(outdir + "/src", "w") as fsrc, open(outdir + "/tgt", "w") as ftgt, open(outdir + "/prompts", "w") as fprompts,  open(outdir + "/scores", "w") as fscore:
        for i in range(len(prompts)):
            tgt_out = json.loads(tgtdata[i])
            all_translations = tgt_out["translations"]
            scores = tgt_out["scores"]
            src_out = srcdata[i]

            for j in range(len(all_translations)):
                fsrc.write(f"{srcdata[i]}\n")
                fprompts.write(f"{prompts[i]}\n")
                ftgt.write(f"{all_translations[j]}\n")
                fscore.write(f"{scores[j]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracts file in a format required by feature extractor ")
    parser.add_argument("--testsrc", help="Name of intput file to sockeye-translate", required=True)
    parser.add_argument("--testtgt", help="Name of output file from sockeye-translate", required=True)
    parser.add_argument("--prompts", help="Ids corresponding to test prompts", required=True)
    parser.add_argument("--outdir", help="Name of desired output file. This will be the shared task format file.", required=True)
    args = parser.parse_args()

    main(args.testsrc, args.testtgt, args.outdir, args.prompts)
