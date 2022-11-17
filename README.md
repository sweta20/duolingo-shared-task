# Duolingo Shared Task 2020

This repository contains the code for our shared task submission: [Generating Diverse Translations via Weighted Fine-tuning and Hypotheses Filtering for the Duolingo STAPLE Task](https://aclanthology.org/2020.ngt-1.21/).


## Training MT models


``` bash

|-- main.sh 									-> Preprocess, Translate, Post-process and evaluate the models
	|-- download-data.sh                    	-> Downloading dataset from OPUS for a given langauge pair 
	|-- process_data.sh 						-> Trains a bpe and tc model and  applies to the dataset
			|-- preprocess.sh	
			|-- unicode_normalize.py			-> Does NFKC normalization on a file
			|-- split-traintest.py    			-> Splits a dataset into train/test/dev
			|-- split-traintest-duo.py 			-> Splits a dataset into train/test/dev based on unique source prompts 
	|-- sockeye-pipeline.sh 					-> Training sockeye models on OPUS data
	|-- sockeye-finetune.sh 					-> Finetune trained model on duolingo
		|-- sockeye-train-transformer.sh        -> Trains a transformer model
		|-- sockeye-train.sh 					-> Trains a attention based model
	
|-- evaluate_all.sh                         	-> Run evaluation for a given language pair 
	|-- preprocess.sh                           -> Preprocess
	|-- sockeye-translate.sh                    -> Translate and Post process
		|-- my-cands-extract.py					-> Converts output of sockeye-translate into shared task format
	|-- evaluate.sh                             -> Run Evaluation scripts
		|-- staple_2020_scorer.py 				-> Official staple scorer from the task	
|-- utils.py 								-> utils file to load shared task data
|-- variables.sh 							-> All global variables and default parameters are declared in this file.

```

You can see all the arguments that main.sh can take by running main.sh -h. For example to run the weighted oversampling model, use:

```
	bash scripts/main.sh -t ja -l weighted -o
```

To evaluate the trained model on test split, 

```
	bash scripts/evaluate_all.sh -s test -t ja -l sockeye-model-finetune-weighted
```

## Scripts for extracting features and training the classifier

``` bash

|-- extract_features_and_rank.sh 				-> Top level file to extract features from hypothesis and run moses reranker or classifer
	|-- run_feature_extractor.sh                -> Run feature extractor on specific files
	    |-- parse-generated-data.py             -> Extracts file in a format required by feature extractor 
		|-- feature-extractor.py                -> Extracts length/ lm and bert scores
		|-- run_eflomal.sh                      -> Used to train/evaluate aligner
		    |-- extract_alignment_score.py      -> Converts into feature format and extracts fertility scores
    |-- generate_topk.py                        -> Extracts top k candidates using threshold extracted from kbmira
    |-- reranker.py                             -> Trains and evaluate model using F1 loss
    	|-- reranker_helper.py                  -> Defines F1 Loss and NN model
|-- train_lm.sh                                 -> Trains a language model on specified dataset using kenlm

```

To train the feature extractor, run the evaluation script on dev split, for eg

```
	bash scripts/evaluate_all.sh -s dev -t ja -l sockeye-model-finetune-weighted 
```

Then, train the feature based classifier using
```
	bash scripts/extract_features_and_rank.sh -s dev -t ja -l sockeye-model-finetune-weighted
```

For filtering:
```
	bash scripts/extract_features_and_rank.sh -s test -t ja -l sockeye-model-finetune-weighted
```


## System Combinations

``` bash

python scripts/combine_lists.py --lista <file1> --listb <file2> --outputfname <outfile>

```

## Create Submission results

The script create_results.sh includes different system combinations we tried for our official dev and test evaluation. 



## Cite our work

If you make use of the code, models, or algorithm, please cite our paper:

```
@inproceedings{agrawal-carpuat-2020-generating,
    title = "Generating Diverse Translations via Weighted Fine-tuning and Hypotheses Filtering for the {D}uolingo {STAPLE} Task",
    author = "Agrawal, Sweta  and
      Carpuat, Marine",
    booktitle = "Proceedings of the Fourth Workshop on Neural Generation and Translation",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.ngt-1.21",
    doi = "10.18653/v1/2020.ngt-1.21",
    pages = "178--187",
    abstract = "This paper describes the University of Maryland{'}s submission to the Duolingo Shared Task on Simultaneous Translation And Paraphrase for Language Education (STAPLE). Unlike the standard machine translation task, STAPLE requires generating a set of outputs for a given input sequence, aiming to cover the space of translations produced by language learners. We adapt neural machine translation models to this requirement by (a) generating n-best translation hypotheses from a model fine-tuned on learner translations, oversampled to reflect the distribution of learner responses, and (b) filtering hypotheses using a feature-rich binary classifier that directly optimizes a close approximation of the official evaluation metric. Combination of systems that use these two strategies achieves F1 scores of 53.9{\%} and 52.5{\%} on Vietnamese and Portuguese, respectively ranking 2nd and 4th on the leaderboard.",
}
```

## Author

Sweta Agrawal
