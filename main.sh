#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;

# Src tgt and BPEs are specified here
src=en
tgt=ja
joint_bpe=False
reverse_dir=False
sub_exp_name=all
oversample=False
topp=1.0
topk=10000
r2l=False
use_tatoeba=False
while getopts "s:t:d:k:b:v:l:z:rejtpoau" opt; do
	case $opt in
		s)
			src=$OPTARG ;;
		t)
			tgt=$OPTARG ;;
		d)
			name=$OPTARG ;;
		v)
			version=$OPTARG ;;
		k)
			topk=$OPTARG ;;
		b)
			bpe_num_operations=$OPTARG ;;
		j)
			joint_bpe=True ;;
		z)
			topp=$OPTARG ;;
		e)
			ensemble=True ;;
		r)
			reverse_dir=True ;;
		p)
			pretokenize=True ;;
		l)
			sub_exp_name=$OPTARG ;;
		o)
			oversample=True;;
		a)
			r2l=True ;;
		u)
			use_tatoeba=True ;;
		h)
			echo "Usage: main.sh"
			echo "-s source_language [default: english]"
			echo "-t target_language [default: Japenese]"
			echo "-d name of the OPUS corpus [default: OpenSubtitles]"
			echo "-v version of the OPUS corpus [default: v2018]"
			echo "-k How many references to use [default: 10000]"
			echo "-z Threshold by LRF value when selecting references [default: 1.0]"
			echo "-v number of merge operations [default: 32000]"
			echo "-j Use joint BPE model for source and target [default: False]"
			echo "-t Use truecasing [default: False]"
			echo "-e Use ensemble [default: False]"
			echo "-p Tokenize the data before using subwords [default: False]"
			echo "-r Train model in reverse direction"
			echo "-o oversample data according to LRF"
			echo "-f Use fairseq pipeline"
			echo "-a Train right to left model"
			echo "-u Use Tatoeba during finetuning"
			exit 0 ;;

		\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1 ;;
		:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1 ;;
	esac
done

# Create global data directory and download dataset if not available
global_data_dir=data/$src-$tgt
if [ ! -d $global_data_dir ]; then
	echo "Downloading data from OPUS"
	. scripts/download-data.sh 
fi;

# Create experiment directory
exp_dir=experiments/$src-$tgt
data_dir=$exp_dir/data
mkdir -p $data_dir

# Create subexperiment directory for duolingo finetuning
tc_model=$data_dir/tc
bpe_model=$data_dir/bpe

echo "Preprocessing data"
duolingo_data_dir=$data_dir/duolingo-$sub_exp_name
mkdir -p $duolingo_data_dir

tatoeba_data_dir=$data_dir/tatoeba
mkdir -p $tatoeba_data_dir

. scripts/process_data.sh

finetune_data_dir=$duolingo_data_dir

. scripts/sockeye-pipeline.sh

if [ $use_tatoeba == True ];then	
	mix_data_dir=$data_dir/${sub_exp_name}
	mkdir -p $mix_data_dir

	for fold in train dev; do
    	for lang in src tgt; do
        	cat $tatoeba_data_dir/$fold.$lang $duolingo_data_dir/$fold-sents.$lang > $mix_data_dir/$fold.$lang
    	done;
	done;
	finetune_data_dir=$mix_data_dir
fi;
echo "Finetuning baseline model"
. scripts/sockeye-finetune.sh
