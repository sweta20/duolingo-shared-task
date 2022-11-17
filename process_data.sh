#!/bin/bash

# Learning models from OPUS data
if [ -f $global_data_dir/$name.$src-$tgt.$src ] && [ -f $global_data_dir/$name.$src-$tgt.$tgt ] && [ ! -f $bpe_model.$src ]; then
    for lang in $src $tgt; do
        python scripts/unicode_normalize.py $global_data_dir/$name.$src-$tgt.$lang

        # Tokenize
        if [ $pretokenize == True ]; then
            if [ ! -f $global_data_dir/$name.tok.$lang ]; then
                echo " * Tokenizing $lang * ..."
                if [ $lang != "ja" ]; then
                    cat $global_data_dir/$name.$src-$tgt.$lang.norm |  $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
                    $moses_scripts/tokenizer/tokenizer.perl -threads 8  -a -l $lang  \
                    > $global_data_dir/$name.tok.$lang
                else
                    $kytea_path -out tok -notags < $global_data_dir/$name.$src-$tgt.$lang.norm > $global_data_dir/$name.tok.$lang;
                fi;
            fi;
        else 
            cat $global_data_dir/$name.$src-$tgt.$lang.norm > $global_data_dir/$name.tok.$lang;
        fi;
    done;

    # Clean dataset
    if [ ! -f $global_data_dir/$name.tok.clean.$lang ]; then
        $moses_scripts/training/clean-corpus-n.perl -ratio 3 $global_data_dir/$name.tok $src $tgt $global_data_dir/$name.tok.clean 1 1000
    fi;
    #Split train/dev/test
    python scripts/split-traintest.py $global_data_dir/$name.tok.clean $src $tgt $data_dir/

    # True-casing
    train=$data_dir/train
    outf=tok.tc
    if [ $truecase == True ]; then
        if [ ! -f $tc_model ]; then
            echo " * Training truecaser using $train.* ..."
            cat $train.$src $train.$tgt > $train.tmp
            $moses_scripts/recaser/train-truecaser.perl \
                -corpus $train.tmp          \
                -model $tc_model
            rm $train.tmp
        fi;

        for type in $src $tgt; do
            cat $train.$type | moses_scripts/recaser/truecase.perl \
            -model $tc_model                       \
            < $train.$type                       \
            > $train.$outf.$type
        done;
    else
        for type in $src $tgt; do
            cat $train.$type | $moses_scripts/tokenizer/lowercase.perl > $train.$outf.$type;
        done;
    fi;

    # BPE models
    inf=$outf
    outf=tok.tc.bpe
    if [ ! -f $bpe_model ]; then
        if [ $joint_bpe == True ]; then
            echo 'Learning BPE jointly'
            if [ $subword_model_type == "bpe" ]; then
                cat $train.$inf.$src $train.$inf.$tgt \
                    | python2 $bpe_scripts_path/learn_bpe.py \
                        -s ${bpe_num_operations} \
                        > $bpe_model.$src
                cp $bpe_model.$src $bpe_model.$tgt
            else
                cat $train.$inf.$src $train.$inf.$tgt > $train.tok.tc.tmp
                python get_sentencepiece.py train \
                 --vocab_size=$bpe_num_operations \
                 --character_coverage=$character_coverage \
                  $train.$inf.tmp $bpe_model.$src 
                cp $bpe_model.$src.model $bpe_model.$tgt.model
                cp $bpe_model.$src.vocab $bpe_model.$tgt.vocab
            fi;
            
        else
            echo "Learning BPE for $src and $tgt separately"
            if [ $subword_model_type == "bpe" ]; then
                cat $train.$inf.$src \
                    | python2 $bpe_scripts_path/learn_bpe.py \
                        -s ${bpe_num_operations} \
                        > $bpe_model.$src

                cat $train.$inf.$tgt \
                    | python2 $bpe_scripts_path/learn_bpe.py \
                        -s ${bpe_num_operations} \
                        > $bpe_model.$tgt
            else
                python get_sentencepiece.py train \
                 --vocab_size=$bpe_num_operations \
                 --character_coverage=$character_coverage \
                  $train.$inf.$src  $bpe_model.$src

                python get_sentencepiece.py train \
                 --vocab_size=$bpe_num_operations \
                 --character_coverage=$character_coverage \
                  $train.$inf.$tgt  $bpe_model.$tgt
            fi;
        fi;
    fi;

fi;

# Applying models to OPUS data
for fold in train dev; do
    for lang in $src $tgt; do
        already_tokenized=True
        data_path=$data_dir/$fold
        if [ $lang == $src ]; then
            type=src
        else
            type=tgt
        fi;
        . scripts/preprocess.sh
    done;
done;

already_tokenized=False


# processing duolingo data
if [ -f ${SHARED_TASK_DATA}/${src}_${tgt}/train.${src}_${tgt}.2020-01-13.gold.txt ]; then
    args=" --topp ${topp} --topk ${topk}"
    if [ $oversample == True ]; then
        args="${args} --oversample "
    fi;
    python scripts/split-traintest-duo.py --fname ${SHARED_TASK_DATA}/${src}_${tgt}/train.${src}_${tgt}.2020-01-13.gold.txt \
     --outputdir $duolingo_data_dir/ --srclang ${src} --tgtlang ${tgt} --extractref ${args}
    args=""
    # python scripts/extract_awsref.py --fname ${SHARED_TASK_DATA}/${src}_${tgt}/train.${src}_${tgt}.aws_baseline.pred.txt --outputdir $duolingo_data_dir/
    for fold in train dev; do
        for lang in $src $tgt; do
            data_path=$duolingo_data_dir/$fold
            if [ $lang == $src ]; then
                type=src
            else
                type=tgt
            fi;
            . scripts/preprocess.sh
        done;
    done;
fi;

name=Tatoeba
python scripts/split-traintest.py $global_data_dir/$name.$src-$tgt $src $tgt $tatoeba_data_dir/ 1500
for fold in train dev; do
    for lang in $src $tgt; do
        data_path=$tatoeba_data_dir/$fold
        if [ $lang == $src ]; then
            type=src
        else
            type=tgt
        fi;
        . scripts/preprocess.sh
    done;
done;
