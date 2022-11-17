#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;


src=en
tgt=vi
split=dev
test_duolingo=True
test_model_name=sockeye-model-finetune-topp_1.0_over
beam_size=10
classify=False
ranker=False
featids="#"
while getopts "s:t:b:m:crf" opt; do
    case $opt in
        s)
            split=$OPTARG ;;
        t)
            tgt=$OPTARG ;;
        b)
            beam_size=$OPTARG;;
        m)
            test_model_name=$OPTARG;;
        c)
            classify=True ;;
        r)
            ranker=True ;;
        f)
            featids=$OPTARG ;;
        h)
            echo "Usage: evaluate_all.sh"
            echo "-s source_language"
            echo "-t target_language"
            exit 0 ;;

    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1 ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1 ;;
    esac
done

global_test_dir=results_duo_${split}/$src-$tgt
python scripts/extract_duo.py --fname ../staple-2020-$split-blind/${src}_${tgt}/$split.${src}_${tgt}.2020-02-20.prompts.txt --outputdir data/${src}-${tgt}/ --name $split
python scripts/extract_awsref.py  --fname ../staple-2020-$split-blind/${src}_${tgt}/$split.${src}_${tgt}.aws_baseline.pred.txt  --outputdir data/${src}-${tgt}/ --prompts data/${src}-${tgt}/$split-duo.prompts --name $split 

# experiment directory
exp_dir=experiments/${src}-${tgt}
data_dir=$exp_dir/data

tc_model=$data_dir/tc
bpe_model=$data_dir/bpe

test_src=data/$src-$tgt/$split-duo.src
test_tgt=data/$src-$tgt/$split-awsref-sents
test_prompts=data/$src-$tgt/$split-duo.prompts

model_dir=$exp_dir/$test_model_name
NBEST=$beam_size
CANDLIMIT=$beam_size
if [ -d $model_dir ]; then
    test_dir=$global_test_dir/$test_model_name/duolingo-$beam_size
    mkdir -p $test_dir
    
    cat $test_src > $test_dir/test.$src
    data_path=$test_dir/test
    lang=$src
    type=src
    . scripts/preprocess.sh

    test_processed=$test_dir/test.src
    . scripts/translate.sh

    if [ ! -f $test_dir/bleu.log ]; then
        args=" "
        if [ $tgt == "ja" ]; then
            args=" ${args} -tok zh "
        fi;

        echo " * Computing bleu* ..." 
        cat $test_dir/sys.out.detok | sacrebleu $args $test_dir/ref.out.detok > $test_dir/bleu.log
        cat $test_dir/bleu.log
    fi;

    outputdir=$test_dir/features/
    test_src=$test_dir/test.src
    mkdir -p $outputdir
    # echo $classify
    if [ $classify == True ] && [ $beam_size == 50 ]; then

        . scripts/run_feature_extractor.sh

        modelpath=features_all/en-$tgt/${test_model_name}-dev/duolingo-$beam_size
        python scripts/reranker.py --featfile $outputdir/features_all --predfile $test_dir/all_cands_detok.txt \
         --modelpath  $modelpath/model --minfile $modelpath/min_val --maxfile $modelpath/max_val --outfile $test_dir/all_cands_filtered.$featids.txt
    fi;

    if [ $ranker == True ] && [ $beam_size == 50 ]; then
        . scripts/run_feature_extractor.sh
        
        modelpath=features_all/en-$tgt/$test_model_name-dev/duolingo-$beam_size/rescore.ini
        python $mosesscorer/rescore.py $modelpath < $outputdir/features_all > $outputdir/features.rescored
        # Get scores
        python scripts/generate_topk.py --featfile $outputdir/features.rescored --predfile $test_dir/all_cands_detok.txt  --outputfile  $test_dir/all_cands_filtered_moses.txt --k 10

    fi;

fi;