import argparse
import numpy as np
from utils import read_trans_prompts, read_transfile, read_file, FIELDSEP
import string
from typing import Dict, List, Set
from collections import Counter
from reranker_helper import F1_Loss, LogisticRegression
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle

table = str.maketrans(dict.fromkeys(string.punctuation))
TRANS_FIELD = 1
FEAT_FIELD = 2
# np.random.seed(0)

def extract_features(feat, featids):
	features = []
	for i in range(len(feat)):
		if not feat[i].startswith("Feature") and feat[i]!="":
			features.append(float(feat[i]))
	if args.featids !="#":
		return [features[i] for i in range(len(features)) if i in list(map(int, [x for x in featids.split("#") if x!=""]))]
	else:
		return features

def read_features(featfile, calculate, minfile, maxfile, featids, normalize):
	features = read_file(featfile)
	feat_dict = {}
	all_feats = []
	for line in features:
		fields = [f.strip() for f in line.split('|||')]
		trans = fields[TRANS_FIELD]
		trans_new = trans.translate(table)
		feats = extract_features(fields[FEAT_FIELD].split(), featids)
		feat_dict[trans_new] = [feats, trans]
		all_feats.append(feats)
	if not normalize:
		return feat_dict
	else:
		if calculate:
			min_val = np.min(np.array(all_feats), axis=0)
			max_val = np.max(np.array(all_feats), axis=0)
			with open(minfile, "wb") as minf, open(maxfile, "wb") as maxf:
				pickle.dump(min_val, minf)
				pickle.dump(max_val, maxf)
		else:
			min_val = pickle.load(open(minfile, "rb"))
			max_val = pickle.load(open(maxfile, "rb"))

		normalize_feat = {}
		for trans in feat_dict: 
			features = np.array(feat_dict[trans][0])
			features = (features-min_val)/ (max_val-min_val)
			normalize_feat[trans] = (features, feat_dict[trans][1])
		return normalize_feat

def read_data(goldfname, predfname, featfile, minfile, maxfile, calculate=False, featids=None, normalize=False):
	with open(predfname, encoding="utf-8") as f:
		lines = f.readlines()
	pred = read_transfile(lines, weighted=False)
	id_text = dict(read_trans_prompts(lines)) 
	feat_dict = read_features(featfile, calculate, minfile, maxfile, featids, normalize )

	if goldfname is not None:
		with open(goldfname, encoding="utf-8") as f:
			lines = f.readlines()
		gold = read_transfile(lines, weighted=True)
		return id_text, gold, pred, feat_dict
	else:
		return id_text, pred, feat_dict

def get_features_and_labels(gold, pred, feat_dict, val=1.0):
	prompt_dict = {}
	for prompt in pred:
		X = []
		Y = []
		Y_weighted = []
		Y_pred = []
		
		ats = sorted(gold[prompt].items(), key=lambda p: p[1], reverse=True)
		paras = []
		sum_v = 0.0
		for k, v in ats:
			paras.append(k)
			sum_v+=v
			if sum_v >= val:
				break
		
		for para, v in pred[prompt].items():
			X.append(feat_dict[para][0])
			Y_pred.append(1.)
			if para in paras:
				Y.append(1.)
				Y_weighted.append(gold[prompt][para])
			else:
				Y.append(0.)
				Y_weighted.append(0.)
		X = np.array(X)
		Y = np.array(Y)
		Y_weighted = np.array(Y_weighted)
		Y_pred = np.array(Y_pred)
		prompt_dict[prompt] = (X, Y_pred, Y, Y_weighted)
	return prompt_dict, X.shape[1]


def train(args):
	_, gold, pred, feat_dict = read_data(args.goldfile, args.predfile, args.featfile, args.minfile, args.maxfile, True, args.featids, args.normalize)
	data_prompts, input_dim = get_features_and_labels(gold, pred, feat_dict)

	output_dim = 2
	hidden_dim = args.hidden
	n_epochs = args.n
	print_every=args.log_every
	device = torch.device("cpu")

	model = LogisticRegression(input_dim, hidden_dim, output_dim)
	model.to(device)
	criterion = F1_Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	logpath = ("/").join(args.modelpath.split("/")[:-1]) + "/log"

	modelpath = args.modelpath + "." + args.featids
	logpath = logpath + "." +  args.featids
	logfile = open(logpath, "w") 

	train_prompts, test_prompts = train_test_split(list(data_prompts.keys()), test_size=0.2)
	best_model = None
	curr_best_f1 = 0.0
	for epoch in range(n_epochs): 
		optimizer.zero_grad()
		loss = 0.0
		random.shuffle(train_prompts)
		for prompt in train_prompts:
			train_X, _, train_Y, train_Y_weighted = data_prompts[prompt]
			train_X = Variable(torch.Tensor(train_X).float())
			train_Y = Variable(torch.Tensor(train_Y).long())
			train_Y_weighted = Variable(torch.Tensor(train_Y_weighted).float())
			out = model(train_X)
			loss += 1 - criterion(out, train_Y, train_Y_weighted).mean()
		loss /= len(train_prompts)
		loss.backward()
		optimizer.step()
		
		if epoch % print_every == 0:
			model.eval()
			with torch.no_grad():
				val_f1 = 0.0
				for prompt in test_prompts:
					test_X, _, test_Y, test_Y_weighted = data_prompts[prompt]
					# wrap up with Variable in pytorch
					test_X = Variable(torch.Tensor(test_X).float())
					test_Y = Variable(torch.Tensor(test_Y).long())
					test_Y_weighted = Variable(torch.Tensor(test_Y_weighted).float())
					out = model(test_X)
					val_f1 += criterion(out,test_Y, test_Y_weighted).mean()
				val_f1 /= len(test_prompts)
				if val_f1 > curr_best_f1:
					best_model = model
					curr_best_f1 = val_f1
			print ('number of epoch', epoch, 'training loss', loss.item(), 'val f1', val_f1.item(), file=logfile)
		model.train()

	torch.save(best_model, modelpath)

def evaluate(args):
	id_text, pred, feat_dict = read_data(args.goldfile, args.predfile, args.featfile, args.minfile, args.maxfile, False, args.featids, args.normalize)
	if args.featids is not None:
		modelpath = args.modelpath + "." + args.featids
	model = torch.load(modelpath)
	model.eval()
	with torch.no_grad(), open(args.outfile, "w") as out :
		for prompt in tqdm(pred):
			src = id_text[prompt]
			out.write(f"\n{prompt}{FIELDSEP}{src}\n")
			for para, v in pred[prompt].items():
				X = feat_dict[para][0]
				test_X = Variable(torch.Tensor(X).float())
				predict_out = model(test_X)
				y_pred = F.softmax(predict_out, dim=0)
				predict_y = torch.argmax(y_pred)
				if predict_y == 1:
					out.write(feat_dict[para][1] + "\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser("This selects threshold based on best scoring k value on dev")
	parser.add_argument("--featfile", help="Path to features file", required=True)
	parser.add_argument("--predfile", help="File to write output to", required=True)
	parser.add_argument("--goldfile", help="Gold annotations", required=False, default=None)
	parser.add_argument("--modelpath", help="if training save to this path else load from this path", required=True)
	parser.add_argument('--train', help='Training', action='store_true')
	parser.add_argument("--hidden", help="hidden dim for 2 layer NN", default=5, type=int)
	parser.add_argument("--lr", help="learning rate", default=0.001, type=float)
	parser.add_argument("--n", help="Number of epochs for training", default=2000, type=int)
	parser.add_argument("--log-every", help="Log loss every x iterations", default=100, type=int)
	parser.add_argument("--outfile", help="Write filtered candidates to")
	parser.add_argument("--minfile", help="Write minval candidates to", default=None)
	parser.add_argument("--maxfile", help="Write maxval candidates to", default=None)
	parser.add_argument('--featids', help="Feature ids to seclude by #", default="#")
	parser.add_argument('--normalize', help='normalize features', action='store_true')

	args = parser.parse_args()

	if args.train:
		train(args)
	else:
		evaluate(args)
