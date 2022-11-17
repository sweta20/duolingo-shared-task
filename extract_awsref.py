import argparse

from utils import read_trans_prompts, read_transfile, FIELDSEP
from sklearn.model_selection import train_test_split
import pickle


def get_data(fname: str, outputdir: str, prompts_fname: str, name: str) -> None:
	"""
	This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)
	For training data, it combines the prompt with all accepted translations. 
	For dev or test data, it combines the prompt only with the most popular translation.
	"""

	with open(fname, encoding="utf-8") as f:
		lines = f.readlines()
	d = read_transfile(lines, strip_punc=False, weighted=False)

	prompts = []
	with open(prompts_fname , encoding="utf-8") as f:
		for line in f:
			prompts.append(line.strip())

	awsfname = outputdir + "/" + name + "-awsref-sents" 
	with open(awsfname, "w") as aws:
		for idstring in prompts:
			ats = d[idstring]

			# make sure that the first element is the largest.
			ats = sorted(ats.items(), key=lambda p: p[1], reverse=True)
			top_ranked_text = ats[0][0]
			print(top_ranked_text, file=aws)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("This extracts aws dataset")
	parser.add_argument("--fname", help="Path of shared task file (probably something like train.en_pt.aws_baseline.pred.txt)", required=True)
	parser.add_argument("--prompts", help="Path to prompts", required=True)
	parser.add_argument("--outputdir", help="Directory to store files", required=True)
	parser.add_argument("--name", help="Name of the output file", default="test")
	args = parser.parse_args()

	get_data(args.fname, args.outputdir, args.prompts, args.name)
