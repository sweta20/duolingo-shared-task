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

def main(srcfile: str, tgtfile: str, outfile: str, sysfile:str, promptsfile: str, candlimit: int, postprocess: bool, model_path: str, fairseq: bool):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 
    """

    prompts = read_file(promptsfile)
    srcdata = read_file(srcfile)
    tgtdata = read_file(tgtfile)

    if postprocess:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path + ".model")

    with open(outfile, "w") as out, open(sysfile, "w") as sys:
        if not fairseq:
            assert len(prompts) == len(srcdata) == len(tgtdata)
            for i in range(len(prompts)):
                tgt_out = json.loads(tgtdata[i])
                if postprocess:
                    all_translations = [sp.decode_pieces(line) for line in tgt_out["translations"]]
                    main_translation = sp.decode_pieces(tgt_out["translation"])
                else:
                    all_translations = tgt_out["translations"]
                    scores = tgt_out["scores"]
                    main_translation = tgt_out["translation"] 

                out.write(f"\n{prompts[i]}{FIELDSEP}{srcdata[i]}\n")
                for j in range(candlimit):
                    out.write(all_translations[j] + "\n")

                sys.write(main_translation + "\n")
        else:
            first = True
            cands = 0
            for line in tgtdata:
                sline = line.strip().split("\t")
                if line.startswith("S-"):
                    textID = prompts[int(sline[0].split("-")[1])]
                    src = srcdata[int(sline[0].split("-")[1])]
                    out.write(f"\n{textID}{FIELDSEP}{src}\n")
                    first = True
                    cands = 0
                elif line.startswith("T-"):
                    # this is the reference
                    sys.write(sline[1] + "\n")
                elif line.startswith("H-"):
                    # this is the prediction, there may be many of these.
                    if candlimit == -1 or cands < candlimit:
                        out.write(sline[2] + "\n")
                        cands += 1
                    # only write the first of these.
                    if first:
                        sys.write(sline[2] + "\n")
                        first = False
                else:
                    pass

    print(f"Wrote to {outfile}, {sysfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This processes the output of sockeye-translate so that it can be scored with sacrebleu and so that it has the shared task format. ")
    parser.add_argument("--testsrc", help="Name of intput file to sockeye-translate", required=True)
    parser.add_argument("--testtgt", help="Name of output file from sockeye-translate", required=True)
    parser.add_argument("--prompts", help="Ids corresponding to test prompts", required=True)
    parser.add_argument("--outfile", help="Name of desired output file. This will be the shared task format file.", required=True)
    parser.add_argument("--sysfile", help="Name of desired output file", required=True)
    parser.add_argument("--candlimit", help="Max number of candidates to put in file (default is -1, meaning all)", type=int, default=-1)
    parser.add_argument('--postprocess', help='Merge sentencepieces using --model ', action='store_true')
    parser.add_argument('--model', help='Merge sentencepieces using --model ')
    parser.add_argument('--fairseq', help='Output in fairseq format', action='store_true')
    args = parser.parse_args()

    main(args.testsrc, args.testtgt, args.outfile, args.sysfile, args.prompts, args.candlimit, args.postprocess, args.model, args.fairseq)
