#!/bin/bash

source /fs/clip-scratch/sweagraw/duolingo/shared-duo/bin/activate

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load cuda/10.0.130                                                    
module load cudnn/7.5.0 

src="en"

type=dev
sub_dir=submissions/results_${type}_$(date +"%Y_%m_%d_%I_%M_%p")/results
mkdir -p $sub_dir


# Best Submission
tgt="pt_br"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results6_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results6_file

model_name=sockeye-model-finetune-duolingo-tatoeba
beam_size=10
results3_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

beam_size=50
results4_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results5_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results3_file --listb $results4_file --outputfname $results5_file

results_file=results_duo_$type/$src-$tgt/sub4_file.txt
python scripts/combine_lists.py --lista $results6_file --listb $results5_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_pt.txt

tgt="vi"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_vi.txt

tgt="hu"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model2_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=10
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_hu.txt

tgt="ja"
model1_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=50
results1_file=results_duo_$type/$src-$tgt/${model1_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model1_name

model2_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

results_file=results_duo_$type/$src-$tgt/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_ja.txt

tgt="ko"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=100
results_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name
cat $results_file > $sub_dir/${src}_ko.txt

# ------------------------------------------------------------------------

# Submission - I : 

declare -A beam_size=( ["pt_br"]=10 ["vi"]=10 ["hu"]=10 ["ko"]=100 ["ja"]=100)
type=dev

model_name=sockeye-model-finetune-topp_1.0_over_all

for tgt in pt_br vi hu ko ja; do
	results_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size[$tgt]}/all_cands_detok.txt
	beam_size=${beam_size[$tgt]}
	bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name
	cat $results_file > $sub_dir/${src}_${tgt}.txt
done;

mv $sub_dir/${src}_pt_br.txt $sub_dir/${src}_pt.txt

# ------------------------------------------------------------------------------- #

# Submission - II : 
#pt
tgt="pt_br"
model1_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model1_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model1_name

model2_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=10
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

results_file=results_duo_$type/$src-$tgt/all_cands_combined_{$model1_name}_{$model2_name}.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_pt.txt

#vi
tgt="vi"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_vi.txt

# ------------------------------------------------------------------------------- #

# Submission - III

#pt
tgt="pt_br"
model1_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model1_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model1_name

model2_name=sockeye-model-finetune-duolingo-tatoeba
beam_size=10
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

beam_size=50
results3_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name -c

results4_file=results_duo_$type/$src-$tgt/${model2_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results3_file --listb $results2_file --outputfname $results4_file

results_file=results_duo_$type/$src-$tgt/all_cands_combined_model1_model2.txt
python scripts/combine_lists.py --lista $results1_file --listb $results4_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_pt.txt

# ------------------------------------------------------------------------------- #

# Submission - IV
# Features  + Alignment features
src="en"

#pt
tgt="pt_br"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results6_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results6_file

model_name=sockeye-model-finetune-duolingo-tatoeba
beam_size=10
results3_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

beam_size=50
results4_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results5_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results3_file --listb $results4_file --outputfname $results5_file

results_file=results_duo_$type/$src-$tgt/sub4_file.txt
python scripts/combine_lists.py --lista $results6_file --listb $results5_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_pt.txt

#vi
tgt="vi"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_vi.txt

# ------------------------------------------------------------------------------- #

# Submission -V

declare -A beam_size=( ["pt_br"]=10 ["vi"]=10 ["hu"]=10 ["ko"]=100 ["ja"]=50)

#pt
for tgt in hu ja; do
	model1_name=sockeye-model-finetune-topp_1.0_over_all
	beam_size=${beam_size[$tgt]}
	results1_file=results_duo_$type/$src-$tgt/${model1_name}/duolingo-${beam_size}/all_cands_detok.txt
	bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model1_name

	model2_name=sockeye-model-finetune-duolingo-tatoeba-all
	beam_size=${beam_size[$tgt]}
	results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
	bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

	results_file=results_duo_$type/$src-$tgt/all_cands_combined.txt
	python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

	cat $results_file > $sub_dir/${src}_${tgt}.txt
done;

# ------------------------------------------------------------------------------------------------

# Submission -VI : Submission IV (pt, vi) + V (hu, ja) + I (ko)

tgt="pt_br"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results6_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results6_file

model_name=sockeye-model-finetune-duolingo-tatoeba
beam_size=10
results3_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

beam_size=50
results4_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results5_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results3_file --listb $results4_file --outputfname $results5_file

results_file=results_duo_$type/$src-$tgt/sub4_file.txt
python scripts/combine_lists.py --lista $results6_file --listb $results5_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_pt.txt


tgt="vi"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_vi.txt


tgt="hu"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model2_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=10
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

results_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_hu.txt

tgt="ja"
model1_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=50
results1_file=results_duo_$type/$src-$tgt/${model1_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model1_name

model2_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model2_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model2_name

results_file=results_duo_$type/$src-$tgt/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results_file

cat $results_file > $sub_dir/${src}_ja.txt

tgt="ko"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=100
results_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name
cat $results_file > $sub_dir/${src}_ko.txt

# ------------------------------------------------------------------------------- #

tgt="pt_br"
model_name=sockeye-model-finetune-topp_1.0_over_all
beam_size=10
results1_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-topp_1.0_over
beam_size=50
results2_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results6_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results1_file --listb $results2_file --outputfname $results6_file

model_name=sockeye-model-finetune-duolingo-tatoeba-all
beam_size=10
results3_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_detok.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name

model_name=sockeye-model-finetune-duolingo-tatoeba
beam_size=50
results4_file=results_duo_$type/$src-$tgt/${model_name}/duolingo-${beam_size}/all_cands_filtered.#.txt
bash scripts/sockeye-predict-blind-data.sh -t $tgt -s $type -b $beam_size -m $model_name -c

results5_file=results_duo_$type/$src-$tgt/${model_name}/all_cands_combined.txt
python scripts/combine_lists.py --lista $results3_file --listb $results4_file --outputfname $results5_file

results_file=results_duo_$type/$src-$tgt/sub4_file.txt
python scripts/combine_lists.py --lista $results6_file --listb $results5_file --outputfname $results_file 

cat $results_file > $sub_dir/${src}_pt.txt

