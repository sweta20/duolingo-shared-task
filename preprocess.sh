
## Variables
# $data_path -> which data to process
# $lang -> language of the data
# $type -> final format of the data
# $pretokenize -> whether to tokenize or not
# $truecase -> whether to trucase or not
# $tc_model -> path to truecase model
# $subword_model_type -> Type of dubword model used for preprocessing
# $bpe_model -> path to subword model to use for preprocessing

outf="tok"
if [ ! -f ${data_path}.$outf.$lang ]; then
    if [ $pretokenize == True ] & [ $already_tokenized == False ]; then
        echo "Tokenizing $data_path.$lang ..."
         if [ $lang != "ja" ]; then
            cat ${data_path}.$lang |  $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
            $moses_scripts/tokenizer/tokenizer.perl -threads 8 -a -l $lang > ${data_path}.$outf.$lang ;
        else
            $kytea_path -out tok -notags < ${data_path}.$lang > ${data_path}.$outf.$lang;
        fi;
    else
        cat ${data_path}.$lang > ${data_path}.$outf.$lang;
    fi;
fi;

inf=$outf
outf="tok.tc"
if [ ! -f ${data_path}.$outf.$lang ]; then
    if [ $truecase == True ]; then
        echo " * True-casing ${data_path}.$inf.$lang ..."
        $moses_scripts/recaser/truecase.perl \
            -model $tc_model                       \
            < ${data_path}.$inf.$lang                      \
            > ${data_path}.$outf.$lang
    else
        cat ${data_path}.$inf.$lang | $moses_scripts/tokenizer/lowercase.perl > ${data_path}.$outf.$lang;
    fi;
fi;

inf=$outf
outf="tok.tc.bpe"
if [ $subword == True ] && [ ! -f ${data_path}.$outf.$lang ]; then
    if [ $subword_model_type == "bpe" ]; then
        python2 $bpe_scripts_path/apply_bpe.py \
            --codes $bpe_model.$lang                  \
            < ${data_path}.$inf.$lang   \
            > ${data_path}.$outf.$lang
    else
        python get_sentencepiece.py encode \
        $bpe_model.$lang ${data_path}.$inf.$lang \
        ${data_path}.$outf.$lang
    fi;
cat ${data_path}.$outf.$lang > ${data_path}.$type
fi;

