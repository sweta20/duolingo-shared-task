#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;

src=en
tgt=vi
split=test
ranker=False
featids="#"
beam_size=50
while getopts "t:d:m:b:k:l:f:r" opt; do
    case $opt in
        t)
            tgt=$OPTARG ;;
		s)
			split=$OPTARG ;;
		m)
			test_model_name=$OPTARG ;;
		b)
			beam_size=$OPTARG ;;
		k)
			k=$OPTARG ;;
		l)
			test_data_name=$OPTARG ;;
		r)
			ranker=True ;;
		f)
			featids=$OPTARG ;;
        h)
            echo "Usage: evaluate_all.sh"
            echo "-s which split to extract features from"
            echo "-t target_language"
            echo "-b beam size"
            echo "-m model name"
            echo "-k threshold for knmira"
            echo "-l dataset to test on"
            echo "-r use mira reranker default[f1 loss]"
            exit 0 ;;

    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1 ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1 ;;
    esac
done

global_test_dir=results/$src-$tgt
#experiment directory
exp_dir=experiments/$src-$tgt
data_dir=$exp_dir/data
duolingo_data_dir=$data_dir/$test_data_name

tc_model=$data_dir/tc
bpe_model=$data_dir/bpe	

global_output_dir=features_all/$src-$tgt
mkdir -p $global_output_dir

for split in dev test; do
	test_prompts=$duolingo_data_dir/$split-prompts
	test_tgt_all=$duolingo_data_dir/$split-all-accepted.$src-$tgt.txt

	test_dir=$global_test_dir/${test_model_name}-$split/duolingo-${beam_size}
	test_src=$test_dir/test.src
	outputdir=$global_output_dir/${test_model_name}-$split/duolingo-${beam_size}
	mkdir -p $outputdir

	echo $outputdir
	. scripts/run_feature_extractor.sh

	if [ $ranker == False ]; then
		if [ $split == dev ]; then
			# if [ ! -f $outputdir/model.$featids ]; then
				python scripts/reranker.py  --train --featfile $outputdir/features_all \
				--predfile $test_dir/all_cands_detok.txt --goldfile $test_tgt_all --modelpath  $outputdir/model \
				--minfile $outputdir/min_val --maxfile $outputdir/max_val --featids $featids
			# fi;
		else
			modelpath=$global_output_dir/${test_model_name}-dev/duolingo-$beam_size
			python scripts/reranker.py --featfile $outputdir/features_all --predfile $test_dir/all_cands_detok.txt \
			--modelpath  $modelpath/model  --featids $featids \
			 --outfile $outputdir/all_cands_filtered.$featids.txt --minfile $modelpath/min_val --maxfile $modelpath/max_val
			python scripts/staple_2020_scorer.py --gold $test_tgt_all --pred $outputdir/all_cands_filtered.$featids.txt > $outputdir/scores.$featids.log
			cat $outputdir/scores.$featids.log
		fi;
	else
		if [ $split == dev ]; then
			refernce=$duolingo_data_dir/$split-sents.$tgt
			# Generate weights
			python $mosesscorer/train.py --nbest $outputdir/features_all --ref $refernce --working-dir $outputdir/ --bin-dir $mosesdecoder/bin

			# Rescore outputs
			python $mosesscorer/rescore.py $outputdir/rescore.ini < $outputdir/features_all > $outputdir/features.rescored

			# Get scores
			python scripts/generate_topk.py --featfile $outputdir/features.rescored --predfile $test_dir/all_cands_detok.txt --goldfile $test_tgt_all --thresholdfile $outputdir/thres
		else
			# Rescore outputs
			python $mosesscorer/rescore.py $global_output_dir/${test_model_name}-dev/duolingo-$beam_size/rescore.ini < $outputdir/features_all > $outputdir/features.rescored
			
			k="$(cat $global_output_dir/${test_model_name}-dev/duolingo-$beam_size/thres)"
			# Get scores
			python scripts/generate_topk.py --featfile $outputdir/features.rescored --predfile $test_dir/all_cands_detok.txt  --outputfile  $outputdir/all_cands_filtered_moses.txt --k 10 >  $outputdir/moses.log
			python scripts/staple_2020_scorer.py --gold $test_tgt_all --pred $outputdir/all_cands_filtered_moses.txt > $outputdir/moses_scores.log
			cat $outputdir/moses_scores.log
		fi;
	fi;
done;