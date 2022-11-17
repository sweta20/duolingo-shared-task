# Score based on staple

if [ $eval_duo == True ]; then
	python `dirname $0`/staple_2020_scorer.py --gold $test_tgt_all --pred $test_dir/all_cands_detok.txt > $test_dir/scores.log
	cat $test_dir/scores.log

fi;

# BLEU
args=" -lc "
if [ $tgt == "ja" ]; then
	args=" ${args} -tok zh "
fi;
echo $args

echo " * Computing bleu* ..." 
if [ ! -f $test_dir/bleu.log ]; then
	cat $test_dir/sys.out.detok | sacrebleu $args $test_dir/ref.out.detok > $test_dir/bleu.log
	cat $test_dir/bleu.log
fi;
