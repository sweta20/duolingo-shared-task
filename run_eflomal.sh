#!/bin/bash
set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh
date;

src=en
tgt=vi
train=False


while getopts "l:td:o:" opt; do
    case $opt in
        l)
            tgt=$OPTARG ;;
		t)
			train=True ;;
		d)
			test_dir=$OPTARG ;;
		o)
			outputdir=$OPTARG ;;
        h)
            echo "Usage: run_eflomal.sh"
            exit 0 ;;

    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1 ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1 ;;
    esac
done

data_dir=experiments/$src-$tgt/data
align_dir=eflomal_alignments/$src-$tgt

if [ $train == True ]; then
	mkdir -p $align_dir
	# All are truecased tokenized
	outf=tok.tc
	train_src=$data_dir/duolingo/train-sents.$outf.$src
	train_tgt=$data_dir/duolingo/train-sents.$outf.$tgt
	cat $data_dir/train.$outf.$src $train_src > $align_dir/train.$src
	cat $data_dir/train.$outf.$tgt $train_tgt > $align_dir/train.$tgt

	N=2000000
	</dev/null paste -d ' ||| ' $align_dir/train.$src - - - - $align_dir/train.$tgt | shuf -n $N  > $align_dir/train.$src-$tgt

	echo "Aligning files"
	$eflomal/align.py -v -i $align_dir/train.$src-$tgt --model 3 -f $align_dir/$src-tgt.fwd -r $align_dir/$src-tgt.rev
	
	echo "Getting Priors"
	$eflomal/makepriors.py -i $align_dir/train.$src-$tgt -f $align_dir/$src-tgt.fwd -r $align_dir/$src-tgt.rev --priors $align_dir/$src-tgt.priors

else
	dev_src=$test_dir/src
	dev_tgt=$test_dir/tgt

	cat $dev_src \
	| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
	> $align_dir/dev.$src

	cat $dev_tgt \
	| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
	> $align_dir/dev.$tgt

	</dev/null paste -d ' ||| ' $align_dir/dev.$src - - - - $align_dir/dev.$tgt > $align_dir/dev.$src-$tgt

	if [ ! -f $outputdir/$src-tgt.dev.rev ]; then
		echo "Aligning files"
		$eflomal/align.py -v -i $align_dir/dev.$src-$tgt --priors $align_dir/$src-tgt.priors --model 3 \
	    -f $outputdir/$src-tgt.dev.fwd -r $outputdir/$src-tgt.dev.rev \
	    -F $outputdir/fwd.scores -R $outputdir/rev.scores
	fi;

    python scripts/extract_alignment_features.py  -i $align_dir/dev.$src-$tgt \
    -f $outputdir/$src-tgt.dev.fwd -r $outputdir/$src-tgt.dev.rev -o $outputdir/align.features

fi;