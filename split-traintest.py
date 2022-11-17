import sys
import random
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

size_test=2000
random.seed(10)
N=10000000

def write_to_file(src_data, tgt_data, src_lang, tgt_lang, output_dir, filename):
	src_file = output_dir + "/" + filename + "." + src_lang
	tgt_file = output_dir + "/" + filename + "." + tgt_lang

	with open(src_file, "w") as f, open(tgt_file, "w") as f1:
		for i in tqdm(range(len(src_data))):
			f.write(src_data[i] )
			f1.write(tgt_data[i])
			

def main():
	filepath = sys.argv[1]
	src_lang = sys.argv[2]
	tgt_lang = sys.argv[3]
	output_dir = sys.argv[4]
	size_test = sys.argv[5]

	src_data = []
	with open(filepath + "." + src_lang) as f:
		for line in f:
			src_data.append(line)

	tgt_data = []
	with open(filepath + "." + tgt_lang) as f:
		for line in f:
			tgt_data.append(line)

	num_of_lines = len(src_data)
	print("Num of lines: ", num_of_lines)

	indices = np.arange(num_of_lines)
	np.random.shuffle(indices)

	src_data_shuffled = [src_data[i] for i in indices[:N]]
	tgt_data_shuffled = [tgt_data[i] for i in indices[:N]]
	print("Len of tgt_data_shuffled: " , len(tgt_data_shuffled))
	
	data = list(zip(src_data, tgt_data))
	random.shuffle(data)
	src_data, tgt_data = zip(*data)

	X_train, X_test, y_train, y_test  = train_test_split(src_data_shuffled, tgt_data_shuffled, test_size=float(size_test)/len(src_data), random_state=10)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=float(size_test)/len(X_train), random_state=10)
	print("Len(train): " , len(X_train), "len(dev): ", len(X_val))

	write_to_file(X_train, y_train, src_lang, tgt_lang, output_dir, "train")
	write_to_file(X_val, y_val, src_lang, tgt_lang, output_dir, "dev")
	write_to_file(X_test, y_test, src_lang, tgt_lang, output_dir, "test")



if __name__ == '__main__':
	main()