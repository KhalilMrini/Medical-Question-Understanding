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
        slines = [sline.split('\t')[-1]]
        for sline in tqdm(source, desc='Processing lines...'):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                                   max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)

                for hypothesis in hypotheses_batch:
                    output_lines.append(hypothesis)
                slines = []

            slines.append(sline.strip().split('\t')[-1])
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_a=args.max_len_a, 
                                           max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
            for hypothesis in hypotheses_batch:
                output_lines.append(hypothesis)

    output_file = open(data_path + args.checkpoint.replace('.pt', '.txt'), 'w')
    input_ids = [line.split('\t')[0] for line in open(data_path + args.source).readlines()]
    for idx, line in enumerate(output_lines):
        print(input_ids[idx] + '\t' + line, file=output_file)
    output_file.close()


parser = argparse.ArgumentParser(description='Test a trained BART model')
parser.add_argument('--model-path', type=str, default='/vault/datasets/khalil/fairseq-rl/checkpoints/', 
                    help='The folder where the model is stored.')
parser.add_argument('--data-path', type=str, default='/vault/datasets/khalil/MeQSum/', 
                    help='The folder where the data is stored.')
parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt', 
                    help='The checkpoint file name.')
parser.add_argument('--source', type=str, default='mediqa2021_test.source', 
                    help='The test file name.')
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