# Preprocess the files

if [ $reverse_dir == True ]; then
	train_src=$data_dir/train.tgt
	train_tgt=$data_dir/train.src
	dev_src=$data_dir/dev.tgt
	dev_tgt=$data_dir/dev.src
	model_name=sockeye-model-run-reverse
elif [ $r2l == True ]; then
	train_src=$data_dir/train.src
	train_tgt=$data_dir/train.tgt.r2l
	dev_src=$data_dir/dev.src
	dev_tgt=$data_dir/dev.tgt.r2l
	model_name=sockeye-model-run-r2l
	if [ ! -f $train_tgt ]; then
		awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' $data_dir/train.tgt > $train_tgt
		awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' $data_dir/dev.tgt > $dev_tgt
	fi;
else 
	train_src=$data_dir/train.src
	train_tgt=$data_dir/train.tgt
	dev_src=$data_dir/dev.src
	dev_tgt=$data_dir/dev.tgt
	model_name=sockeye-model-run
fi;

## pipeline parameters
avg_metric_list="perplexity bleu"
avg_n_list="1 4 8"
if [[ $ensemble == True ]]; then
	run_n=4
else
	run_n=1
fi;

for i in $(seq 1 $run_n); do
	model_dir=$exp_dir/${model_name}-$i
	## Training model-$i
	seed=$i
	echo "Training the model"
	. `dirname $0`/sockeye-train-transformer.sh
done;

date;
