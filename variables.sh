#!/bin/bash

source /fs/clip-scratch/sweagraw/duolingo/shared-duo/bin/activate

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load cuda/10.0.130                                                    
module load cudnn/7.5.0 

# Change this to the location where you downloaded the data.
root_dir=/fs/clip-scratch/sweagraw/duolingo/corpora
SHARED_TASK_DATA=/fs/clip-scratch/sweagraw/duolingo/staple-2020-train
moses_scripts=/fs/clip-scratch/sweagraw/multitask/moses-scripts
bpe_scripts_path=/fs/clip-scratch/sweagraw/multitask/subword-nmt/subword_nmt
sacrebleu_path=/fs/clip-scratch/sweagraw/multitask/sockeye/contrib/sacrebleu
kenlm_path=/fs/clip-scratch/sweagraw/software/kenlm/build/bin
kytea_path=/fs/clip-scratch/software/kytea/bin/kytea
mosesdecoder=/fs/clip-sw/user-supported/mosesdecoder/3.0
mosesscorer=$moses_scripts/nbest-rescore
gizapp=/fs/clip-scratch/sweagraw/software/giza-pp/GIZA++-v2
berkeleyaligner=/fs/clip-software/user-supported/berkeleyaligner/unsupervised-2.1/berkeleyaligner
eflomal=/fs/clip-scratch/sweagraw/software/eflomal  

# Default baseline dataset
name=OpenSubtitles
truecase=False
version=v2018

# Default locations for tc and bpe folders
truecase=False
subword=True
pretokenize=True
bpe_num_operations=32000
subword_model_type=bpe
character_coverage=1.0

# By default duo_learn_bpe_tc is False as we are not learning a tc or BPE model for only duolingo data
duo_learn_bpe_tc=False
duo_bpe_num_operations=20000

## pipeline training parameters
src_max_len=80
tgt_max_len=80
ensemble=False

gpus=0
proc_per_gpu=1
transformer=True
eval=False

# Evaluation
gpu_id=0
eval=True
already_tokenized=False #Since for opensubtitles the split is created after tokenizing text.
fairseq=False
use_diverse=-1
sample=False