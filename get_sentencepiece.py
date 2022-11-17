#!/usr/bin/env python
"""
Usage:
    get_sentencepiece.py train [options] INPUT_FILE MODEL_PATH
    get_sentencepiece.py encode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    get_sentencepiece.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE

Options:
    -h --help                               Show this screen.
    --vocab_size=<int>                      vocab size [default: 32000]
    --character_coverage=<float>            character coverage [default: 1.0]
"""

import sentencepiece as spm
from docopt import docopt

def train(args):
    spm.SentencePieceTrainer.Train('''--hard_vocab_limit=false --input={0} --model_prefix={1} --vocab_size={2} --character_coverage={3}'''.format(args['INPUT_FILE'],
         args['MODEL_PATH'], int(args['--vocab_size']), float(args['--character_coverage'])))

def encode(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args['MODEL_PATH'] + ".model")

    with open(args['OUTPUT_FILE'], 'w', encoding='utf-8') as f1:
        with open(args['TEST_SOURCE_FILE'], encoding='utf-8') as f:
            for line in f:
                encodings = sp.EncodeAsPieces(line)
                f1.write((" ").join(encodings) + "\n")

def decode(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args['MODEL_PATH']  + ".model")

    with open(args['OUTPUT_FILE'], 'w', encoding='utf-8') as f1:
        with open(args['TEST_SOURCE_FILE'], encoding='utf-8') as f:
            for line in f:
                decodings = sp.decode_pieces(line)
                f1.write((" ").join(decodings) + "\n")
                

def main():
    args = docopt(__doc__)

    # seed the random number generators
    if args['train']:
        train(args)
    elif args['encode']:
        encode(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()
