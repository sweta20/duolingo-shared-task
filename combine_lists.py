import argparse

from utils import read_trans_prompts, read_transfile, write_to_file
import pickle

def combine_lists(list_a, list_b, method="union"):
	pred_all = {}
	for prompt in list_a:
		para_pred_a = list_a[prompt].keys() if prompt in list_a else {}
		para_pred_b = list_b[prompt].keys() if prompt in list_b else {}
		if method =="union":
			pred_all[prompt] = {k: 1 for k in set(para_pred_a).union(para_pred_b)}
		elif method =="intersection":
			pred_all[prompt] = {k: 1 for k in set(para_pred_a).intersection(para_pred_b)}
	return pred_all

def get_data(flista: str, flistb: str, method: str, outputfname: str) -> None:
	with open(flista, encoding="utf-8") as f:
		lines = f.readlines()
	list_a = read_transfile(lines, strip_punc=True, weighted=False)
	id_text = dict(read_trans_prompts(lines)) 
	
	with open(flistb, encoding="utf-8") as f:
		lines = f.readlines()
	list_b = read_transfile(lines, strip_punc=True, weighted=False)

	list_com = combine_lists(list_a, list_b, method)
	write_to_file(list_com, id_text, outputfname)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("This combines outputs from multiple files")
	parser.add_argument("--lista", help="Path to cands file 1", required=True)
	parser.add_argument("--listb", help="Path to cands file 2", required=True)
	parser.add_argument("--method", help="Path to prompts", default="union")
	parser.add_argument("--outputfname", help="Directory to store files", required=True)
	args = parser.parse_args()

	get_data(args.lista, args.listb, args.method, args.outputfname)
