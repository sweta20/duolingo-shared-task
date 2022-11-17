#!/bin/bash

# Params outputdir, src, tgt, test_dir

# Extract candidates in a format required for extracting features
if [ ! -f $test_dir/src ] ; then
	python scripts/parse-generated-data.py \
		 --testsrc  $test_src \
		--testtgt  $test_dir/gen.out \
		--outdir $test_dir \
		--prompts $test_prompts 
fi; 
# Post-processing translations
if [ ! -f $outputdir/src.detok ]; then
	cat $test_dir/src \
		| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
		| $moses_scripts/tokenizer/detokenizer.perl -q -l $src 2>/dev/null \
		> $outputdir/src.detok
fi;

if [ ! -f $outputdir/tgt.detok ]; then
	cat $test_dir/tgt \
		| sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
		| $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt 2>/dev/null \
		> $outputdir/tgt.detok	
fi;

if [ ! -f $outputdir/tgt.r2l ]; then
	awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' $test_dir/tgt > $outputdir/tgt.r2l 
fi;

if [ ! -f $outputdir/features_all ]; then
	# All features: length; lm; bert
	if [ ! -f $outputdir/features ]; then
		python scripts/feature-extractor.py  --srcfile $outputdir/src.detok --tgtfile $outputdir/tgt.detok --srclang $src --tgtlang $tgt --outfile $outputdir/features \
		--lm  $tgt/Open_Duo_W/Open_Duo_W.lm.bin
	fi;

	# MT features + R2L + Reconstruction features
	sed -i 's/$/ Feature3= /'  $outputdir/features 
	paste -d' ' $outputdir/features $test_dir/scores > $outputdir/features_mod  && mv $outputdir/features_mod $outputdir/features
	for model_name in sockeye-model-run-1 sockeye-model-run-reverse-1 sockeye-model-run-r2l-1; do
	 	if [ ! -f $outputdir/$model_name.score ] && [ -d $exp_dir/$model_name/  ]; then
	 		# --score-type logprob (normalized) --output-type  pair_with_score
	 		if [[  "$model_name" == *"reverse"* ]]; then
	 			python3 -m sockeye.score -m $exp_dir/$model_name/ --source $test_dir/tgt --target $test_dir/src --score-type neglogprob > $outputdir/$model_name.score
	 		elif [[  "$model_name" == *"r2l"* ]]; then
				python3 -m sockeye.score -m $exp_dir/$model_name/ --source $test_dir/src --target $outputdir/tgt.r2l --score-type neglogprob > $outputdir/$model_name.score
	 		else
	 			python3 -m sockeye.score -m $exp_dir/$model_name/ --source $test_dir/src --target $test_dir/tgt --score-type neglogprob > $outputdir/$model_name.score
	 		fi;
	 	paste -d' ' $outputdir/features $outputdir/$model_name.score > $outputdir/features_mod  && mv $outputdir/features_mod $outputdir/features
	 	fi;
	 	# rm  $outputdir/$model_name.score ;
	done;


	sed -i 's/$/ Feature4= /'  $outputdir/features
	bash scripts/run_eflomal.sh -l $tgt -d $test_dir -o $outputdir
	paste -d' ' $outputdir/features $outputdir/fwd.scores > $outputdir/features_mod  && mv $outputdir/features_mod $outputdir/features
	paste -d' ' $outputdir/features $outputdir/rev.scores > $outputdir/features_mod  && mv $outputdir/features_mod $outputdir/features
	paste -d' ' $outputdir/features $outputdir/align.features > $outputdir/features_mod  && mv $outputdir/features_mod $outputdir/features
	sed -i 's/$/ \|\|\| /'  $outputdir/features 

	cat $outputdir/features > $outputdir/features_all
	rm $outputdir/features
fi;