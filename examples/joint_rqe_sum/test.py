import argparse
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np

def test(args):
    model_path = args.model_path #'./results/cnn-dm-rl/'
    data_path = args.data_path #'../cnn-dm-data-bin/'

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bart = BARTModel.from_pretrained(model_path, checkpoint_file=args.checkpoint, data_name_or_path=data_path[:-1] + '-bin/')#"/vault/datasets/khalil/fairseq-rl/models/bart.large.xsum")##

    bart = bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = 1

    output_lines = []

    with open(data_path + args.source) as source:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source, desc='Processing lines...'):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                                   max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)

                for hypothesis in hypotheses_batch:
                    output_lines.append(hypothesis)
                slines = []
                
            if count == 50:
                break

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                           max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
            for hypothesis in hypotheses_batch:
                output_lines.append(hypothesis)

    rougel = []
    rougelp = []
    rougelr = []
    rouge1 = []
    rouge1p = []
    rouge1r = []
    rouge2 = []
    rouge2p = []
    rouge2r = []
    target_lines = [line.strip() for line in open(data_path + args.target).readlines()]
    idx = 0
    src_lines = open(data_path + args.source).readlines()
    for pred, tgt in tqdm(zip(output_lines, target_lines), desc='Scoring...'):
        print('SOURCE', src_lines[idx])
        print('PRED', pred)
        print('TGT', tgt)
        score = scorer.score(tgt, pred)
        rougel.append(score['rougeL'].fmeasure)
        rougelp.append(score['rougeL'].precision)
        rougelr.append(score['rougeL'].recall)
        rouge1.append(score['rouge1'].fmeasure)
        rouge1p.append(score['rouge1'].precision)
        rouge1r.append(score['rouge1'].recall)
        rouge2.append(score['rouge2'].fmeasure)
        rouge2p.append(score['rouge2'].precision)
        rouge2r.append(score['rouge2'].recall)
        idx += 1
    
    print('Rouge-1:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rouge1), np.mean(rouge1p), np.mean(rouge1r)))
    print('Rouge-2:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rouge2), np.mean(rouge2p), np.mean(rouge2r)))
    print('Rouge-L:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rougel), np.mean(rougelp), np.mean(rougelr)))


parser = argparse.ArgumentParser(description='Test a trained BART model')
parser.add_argument('--model-path', type=str, default='/vault/datasets/khalil/fairseq-rl/checkpoints/', 
                    help='The folder where the model is stored.')
parser.add_argument('--data-path', type=str, default='/vault/datasets/khalil/MeQSum/', 
                    help='The folder where the data is stored.')
parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt', 
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