#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;

already_tokenized=False
subword=False
while getopts "b:c:o:l:ps" opt; do
	case $opt in
		b)
			bpe_model=$OPTARG ;;
		c)
			corpus=$OPTARG ;;
		o)
			name=$OPTARG ;;
		l)
			lang=$OPTARG ;;
		p)
			already_tokenized=True ;;
		s)
			subword=True ;;
		h)
			echo "Usage: main.sh"
			echo "-s source_language [default: english]"
			echo "-t target_language [default: portuguese]"
			echo "-c corpus [eg $data_dir/train.pt]"
			exit 0 ;;

		\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1 ;;
		:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1 ;;
	esac
done

lm_dir=language_models/$lang/$name
lm_data_dir=$lm_dir/data
mkdir -p $lm_data_dir
lm=$lm_dir/$name.lm


# Training language models
# cat $corpus > $lm_data_dir/$name.$lang
# data_path=$lm_data_dir/$name
# type=src
# . scripts/preprocess.sh
outf=tok.tc

exp_dir=experiments/en-$lang
data_dir=$exp_dir/data
train_tgt=$data_dir/duolingo-weighted/train-sents.$outf.$lang
cat $data_dir/train.$outf.$lang $train_tgt> $lm_data_dir/train.$lang


# n=47000000
# head -n $n ${data_path}.$outf.$lang > ${data_path}.$outf.$lang.$n

# Training language models
if [ ! -f $lm.bin ]; then
	echo " * Training LM for ${corpus} ..."
	cat $lm_data_dir/train.$lang \
		| $kenlm_path/lmplz --order 5 -S 20G -T $lm_dir/ \
		> $lm
    $kenlm_path/build_binary $lm $lm.bin
	rm $lm
fi;