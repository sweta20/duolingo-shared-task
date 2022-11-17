# Translate and postprocess files
# test_dir
# NBEST
# gpu_id
# CANDLIMIT
# beam_size


if [ ! -f $test_dir/gen.out ]; then
	if [ $eval_duo == True ]; then

		sockeye_args=""
		if [ $sample == True ]; then
			sockeye_args=" ${sockeye_args} --sample 1000 "
		fi;

		python3 -m sockeye.translate    \
			--input $test_processed   \
			--output $test_dir/gen.out \
			--models $model_dir  \
			--ensemble-mode linear \
			--batch-size 1  \
			--chunk-size 1024 \
			--device-ids $gpu_id \
			--nbest-size ${NBEST} \
			 --beam-size ${beam_size} \
			--output-type translation_with_score \
			$sockeye_args \
			--disable-device-locking
	else
		python3 -m sockeye.translate    \
			--input $test_processed   \
			--output $test_dir/sys.out \
			--models $model_dir  \
			--ensemble-mode linear \
			--beam-size 5   \
			--batch-size 16  \
			--chunk-size 1024 \
			--device-ids $gpu_id \
			--disable-device-locking
	fi;
fi;


args=""
# Post process data
if [ $subword_model_type == "sp" ]; then
	args=" ${args} --postprocess --model $bpe_model.$tgt "
fi;

if [ $fairseq == True ]; then
	args=" ${args} --fairseq "
fi;

# Extract candidates in a format required by duolingo and create a best prediction file
if [ ! -f $test_dir/sys.out ] && [ ! -f $test_dir/all_cand.txt ] ; then
	python `dirname $0`/my-cands-extract.py --testsrc  $test_src \
		--testtgt  $test_dir/gen.out \
		--outfile $test_dir/all_cands.txt \
		--sysfile $test_dir/sys.out \
		--prompts $test_prompts  \
		--candlimit $CANDLIMIT \
		${args}
fi;

# Post-processing translations
if [ ! -f $test_dir/sys.out.detok ]; then
	echo " * Post-processing $test_dir/test.trans ..."
	if [ $truecase == True ]; then
		cat $test_dir/sys.out \
			| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
			| $moses_scripts/recaser/detruecase.perl 2>/dev/null \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/sys.out.detok
	else
		cat $test_dir/sys.out \
			| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/sys.out.detok
	fi;
fi;

if [ -f $test_tgt ] && [ ! -f $test_dir/ref.out.detok ]; then 
	echo " * Post-processing $test_tgt ..."
	if [ $truecase == True ]; then
		cat $test_tgt \
			| $moses_scripts/recaser/detruecase.perl 2>/dev/null \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/ref.out.detok
	else
		cat $test_tgt \
			| $moses_scripts/tokenizer/lowercase.perl \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/ref.out.detok
	fi;
fi;

if [ ! -f $test_dir/all_cands_detok.txt ] && [ -f $test_dir/all_cands.txt ]; then
	echo " * Post-processing $test_dir/all_cand.txt ..."
	if [ $truecase == True ]; then
		cat $test_dir/all_cands.txt \
			| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
			| $moses_scripts/recaser/detruecase.perl 2>/dev/null \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/all_cands_detok.txt
	else
		cat $test_dir/all_cands.txt \
			| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
			| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
			> $test_dir/all_cands_detok.txt
	fi;
fi;
