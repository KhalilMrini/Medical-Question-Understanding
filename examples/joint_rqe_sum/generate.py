import argparse
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import random

def test(args):
    model_path = args.model_path #'./results/cnn-dm-rl/'
    data_path = args.data_path #'../cnn-dm-data-bin/'

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bart = BARTModel.from_pretrained(model_path, checkpoint_file=args.checkpoint, data_name_or_path=data_path[:-1] + '-bin/')

    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = 32

    output_lines = []
    output_lines2 = []

    src_lines = [line.strip() for line in open(data_path + args.source).readlines()]
    tgt_lines = [line.strip() for line in open(data_path + args.target).readlines()]
    input_lines = list(zip(src_lines, tgt_lines))
    input_lines = random.choices(input_lines, k=40)

    slines = [input_lines[0][0]]
    for sline in tqdm(input_lines[1:], desc='Processing lines...'):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                                max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)

            for hypothesis in hypotheses_batch:
                output_lines.append(hypothesis)
            slines = []

        slines.append(sline[0])
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                        max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
        for hypothesis in hypotheses_batch:
            output_lines.append(hypothesis)

    rougel = []
    rouge1 = []
    rouge2 = []
    for pred, tgt in tqdm(zip(output_lines, input_lines), desc='Scoring...'):
        score = scorer.score(tgt[1], pred)
        rougel.append(score['rougeL'])
        rouge1.append(score['rouge1'])
        rouge2.append(score['rouge2'])
    
    print('checkpoint1, RougeL: {}\nRouge1: {}\nRouge2: {}'.format(np.mean(rougel), np.mean(rouge1), np.mean(rouge2)))

    bart = BARTModel.from_pretrained(model_path, checkpoint_file=args.checkpoint2, data_name_or_path=data_path[:-1] + '-bin/')

    slines = [input_lines[0][0]]
    for sline in tqdm(input_lines[1:], desc='Processing lines...'):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                                max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)

            for hypothesis in hypotheses_batch:
                output_lines2.append(hypothesis)
            slines = []

        slines.append(sline[0])
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                        max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
        for hypothesis in hypotheses_batch:
            output_lines2.append(hypothesis)

    rougel = []
    rouge1 = []
    rouge2 = []
    for pred, tgt in tqdm(zip(output_lines2, input_lines), desc='Scoring...'):
        score = scorer.score(tgt[1], pred)
        rougel.append(score['rougeL'])
        rouge1.append(score['rouge1'])
        rouge2.append(score['rouge2'])
    
    print('checkpoint2, RougeL: {}\nRouge1: {}\nRouge2: {}'.format(np.mean(rougel), np.mean(rouge1), np.mean(rouge2)))

    with open('/vault/datasets/khalil/fairseq-rl/' + data_path.split('/')[-2] + '.txt', 'w') as output:

        for input_, pred, pred2 in tqdm(zip(input_lines, output_lines, output_lines2), desc='Printing'):
            print('\t'.join([input_[0], input_[1], pred, pred2]), file=output)


parser = argparse.ArgumentParser(description='Test a trained BART model')
parser.add_argument('--model-path', type=str, default='/vault/datasets/khalil/fairseq-rl/checkpoints/', 
                    help='The folder where the model is stored.')
parser.add_argument('--data-path', type=str, default='/vault/datasets/khalil/MeQSum/', 
                    help='The folder where the data is stored.')
parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt', 
                    help='The checkpoint file name.')
parser.add_argument('--checkpoint2', type=str, default='checkpoint_best.pt', 
                    help='The checkpoint file name.')
parser.add_argument('--source', type=str, default='test.source', 
                    help='The test file name.')
parser.add_argument('--target', type=str, default='test.target', 
                    help='The output file name.')
parser.add_argument('--beam', type=int, default=4, 
                    help='The beam search size.')
parser.add_argument('--lenpen', type=float, default=2., 
                    help='The length penalty.')
parser.add_argument('--max-len-a', type=int, default=0)
parser.add_argument('--max-len-b', type=int, default=40)
parser.add_argument('--min-len', type=int, default=5)
parser.add_argument('--no-repeat-ngram-size', type=int, default=3)

args = parser.parse_args()
test(args)