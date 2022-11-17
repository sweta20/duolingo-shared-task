import argparse  
import numpy as np
from tqdm import tqdm
from mosestokenizer import *
import kenlm
from bert_score import score
from bert_score import plot_example
from bert_score import BERTScorer
FIELDSEP = " ||| "
lm_dir="/fs/clip-scratch/sweagraw/duolingo/corpora/language_models/"
import stanza

import math
ln10 = math.log(10)
class StanzaModel():
    def __init__(self,lang):
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang)
    
    def get_parse(self, text):
        doc = self.nlp(text)
        deprels = []
        pos_tags = []
        count = 0
        for sentence in doc.sentences:
            for word in sentence.words:
                deprels.append(word.deprel)
                pos_tags.append(word.pos)
                count+=1
        return deprels, pos_tags, count

def read_file(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

class FeatureExtractor():
    def __init__(self, config): #dictionary
        src_lang = config["srclang"]
        tgt_lang = config["tgtlang"]
        tgt_lang = "pt" if tgt_lang == "pt_br" else tgt_lang
        bert_model_name = config["bert_model_name"]
        lm_models_paths = config["lm"] #list of paths to lms
        
        self.src_tokenize = MosesTokenizer(src_lang)
        self.tgt_tokenize = MosesTokenizer(tgt_lang)

        # self.src_stanza = StanzaModel(src_lang)
        # self.tgt_stanza = StanzaModel(tgt_lang)
        
        self.lm_models = []
        for path in lm_models_paths:
            self.lm_models.append(kenlm.Model(path))
            
        self.scorer = BERTScorer(model_type=bert_model_name)
        
    def get_length(self, text):
        return len(text.split(" "))

    def lm_scores(self, text, length):
        # Normaliza LM score by length?
        scores = []
        for lm in self.lm_models:
            # highest_n_gram_score, highest_n_gram, _ = sorted(lm.full_scores(text), key=lambda p: p[1], reverse=True)[0]
            lm_score = lm.score(text)
            log_scaled = round(lm_score*ln10,4)
            scores.extend([(log_scaled * 1.0 ) / length])
        return np.array(scores)

    def shallow_depmatch(self, text1, text2):
        src_deps, _, _ = self.src_stanza.get_parse(text1)
        tgt_deps, _, tgt_len = self.tgt_stanza.get_parse(text2)
        return [len(set(src_deps).intersection(tgt_deps))/tgt_len]

    def bert_score(self, text1, text2): # wrt to Refernce with highest LRF -> dont have this at tgt?
        p ,r,f = self.scorer.score([text2], [text1])
        return [p.cpu().numpy().item(), r.cpu().numpy().item(), f.cpu().numpy().item()]
    
    def extract_features(self, src, tgt):
        tokenized_src = (" ").join(self.src_tokenize(src))
        tokenized_tgt = (" ").join(self.tgt_tokenize(tgt))
        len_src = self.get_length(tokenized_src)
        len_tgt = self.get_length(tokenized_tgt)
        return [[len_src, len_tgt], self.lm_scores(tokenized_tgt,  self.get_length(tokenized_tgt))]
        # return [[len_src, len_tgt, float(len_src)/len_tgt, float(len_tgt)/len_src], self.shallow_depmatch(src, tgt), self.bert_score(src, tgt), self.lm_scores(tokenized_tgt,  self.get_length(tokenized_tgt))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)")
    parser.add_argument("--srcfile", help="Path of shared task src file", required=True)
    parser.add_argument("--tgtfile", help="Path of shared task tgt file", required=True)
    parser.add_argument("--srclang", help="Name of desired src language, probably something like en", required=True)
    parser.add_argument("--tgtlang", help="Name of desired tgt language, probably something like vi", required=True)
    parser.add_argument("--outfile", help="File to write features to", required=True)
    parser.add_argument("--lm", help="name of language models", nargs='+', required=True)
    parser.add_argument("--bert-model-name", help="Bert model fot bert-score", default="xlm-mlm-100-1280")
    args = parser.parse_args()

    config = {
    "srclang" : args.srclang,
    "tgtlang" : args.tgtlang,
    "lm" : [lm_dir + x for x in  args.lm],
    "bert_model_name" : args.bert_model_name  # bert-base-multilingual-cased
    }

    src_text = read_file(args.srcfile)
    tgt_text = read_file(args.tgtfile)

    assert len(src_text) == len(tgt_text) 
    feat_extract = FeatureExtractor(config)

    with open(args.outfile, "w", encoding="utf-8") as out:
        k=0
        for i in tqdm(range(len(src_text))):
            features = feat_extract.extract_features(src_text[i], tgt_text[i])
            feature_text = ""
            for t in range(len(features)):
                feature_text+=" Feature" + str(t) +"= "
                feature_text+=(" ").join([str(x) for x in features[t]])
            
            out.write(f"{k}{FIELDSEP}{tgt_text[i]}{FIELDSEP}{feature_text}\n")
            if i+1!=len(src_text) and src_text[i+1] != src_text[i]:
                k+=1

            # out.write(f"{src_text[i]}{FIELDSEP}{tgt_text[i]}{FIELDSEP}{(FIELDSEP).join([str(x) for x in feat_extract.extract_features(src_text[i], tgt_text[i])])}\n")
