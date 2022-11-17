import argparse

from utils import read_trans_prompts, read_transfile, FIELDSEP
from sklearn.model_selection import train_test_split
import pickle


def get_data(fname: str, outputdir: str, name:str) -> None:
    """
    This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)
    For training data, it combines the prompt with all accepted translations. 
    For dev or test data, it combines the prompt only with the most popular translation.
    """

    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    duo_src = outputdir + "/" + name  +"-duo.src"
    prompts = outputdir + "/" + name  +"-duo.prompts"

    with open(duo_src, "w") as s, open(prompts, "w") as p:
        for line in lines:
            # print(line)
            key, prompt = line.strip().lower().split(FIELDSEP)
            print(key, file=p)
            print(prompt, file=s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracts dev data")
    parser.add_argument("--fname", help="Path of shared task file (probably something like dev.en_vi.2020-01-13.gold.txt)", required=True)
    parser.add_argument("--outputdir", help="Directory to store files", required=True)
    parser.add_argument("--name", help="Name of the output file", required=True)
    args = parser.parse_args()

    get_data(args.fname, args.outputdir, args.name)
