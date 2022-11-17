#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;

src=en
tgt=ja
split=test
# default
test_duo_dir=duolingo
eval_duo=False

while getopts "s:t:u:n:l:d:po" opt; do
    case $opt in
        s)
            split=$OPTARG ;;
        t)
            tgt=$OPTARG ;;
		d)
			test_duo_dir=$OPTARG ;;
		n)
			name=$OPTARG ;;
		o)
			eval_duo=True;;
		l)
			sub_exp_name=$OPTARG ;;
		p)
			sample=True ;;
        h)
            echo "Usage: evaluate_all.sh"
            echo "-s split [dev/test]"
            echo "-t target_language"
            echo "-l which model directory to use"
            echo "-p sample candidates instead of using beam search"
            echo "-d test on duolingo dataset using official scorer "
            exit 0 ;;

    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1 ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1 ;;
    esac
done

# Todo: Add global data src: Not optimized for memory :()
global_test_dir=results/$src-$tgt
mkdir -p $global_test_dir

#experiment directory
exp_dir=experiments/$src-$tgt
data_dir=$exp_dir/data
duolingo_data_dir=$data_dir/$test_duo_dir

tc_model=$data_dir/tc
bpe_model=$data_dir/bpe

test_src=$duolingo_data_dir/$split-sents.$src
test_tgt=$duolingo_data_dir/$split-sents.$tgt
test_prompts=$duolingo_data_dir/$split-prompts
test_tgt_all=$duolingo_data_dir/$split-all-accepted.$src-$tgt.txt
test_model_name=$sub_exp_name


model_dir=$exp_dir/$test_model_name
for beam_size in 10 50; do
	CANDLIMIT=${beam_size}
	NBEST=$beam_size
	if [ -d $model_dir ]; then
		test_dir=$global_test_dir/${test_model_name}-$split/duolingo-$beam_size
		if [ $sample == True ]; then
			test_dir=${test_dir}_sample
		fi;

		mkdir -p $test_dir

		cat $test_src > $test_dir/test.$src
		data_path=$test_dir/test
		lang=$src
		type=src
		. scripts/preprocess.sh

		test_processed=$test_dir/test.src
		. scripts/translate.sh

		. scripts/evaluate.sh
	fi;
done;

