#!/bin/bash


if [ $r2l == True ]; then
	train_src=$finetune_data_dir/train.src
	train_tgt=$finetune_data_dir/train.tgt.r2l
	dev_src=$finetune_data_dir/dev.src
	dev_tgt=$finetune_data_dir/dev.tgt.r2l
	model_name=sockeye-model-finetune-r2l
	awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' $finetune_data_dir/train.tgt > $train_tgt.r2l
	awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' $finetune_data_dir/dev.tgt > $dev_tgt.r2l
else 
	train_src=$finetune_data_dir/train.src
	train_tgt=$finetune_data_dir/train.tgt
	dev_src=$finetune_data_dir/dev.src
	dev_tgt=$finetune_data_dir/dev.tgt
	model_name=sockeye-model-finetune
fi;

# echo $train_src

model_dir=$exp_dir/${model_name}-${sub_exp_name}

seed=1
# By default I am finetuning sockeye-model-run-1. Need to parameterize it!
model_args=" --params experiments/$src-$tgt/sockeye-model-run-1/params.best \
	--source-vocab experiments/$src-$tgt/sockeye-model-run-1/vocab.src.0.json \
	--target-vocab experiments/$src-$tgt/sockeye-model-run-1/vocab.trg.0.json"

echo "Training the model"
. `dirname $0`/sockeye-train-transformer.sh


date;
