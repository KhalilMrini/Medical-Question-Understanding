import argparse
import torch
from fairseq.models.bart import BARTModel
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from rouge_score import rouge_scorer
from torch.nn.functional import cosine_similarity
import numpy as np

def test(args):
    data_path = args.data_path #'../cnn-dm-data-bin/'

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    model = BertModel.from_pretrained("bert-base-cased").cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    output_lines = [line.strip() for line in open(args.source).readlines()]

    rougel = []
    rougelp = []
    rougelr = []
    rouge1 = []
    rouge1p = []
    rouge1r = []
    rouge2 = []
    rouge2p = []
    rouge2r = []
    bert_p = []
    bert_r = []
    bert_f = []
    target_lines = [line.strip() for line in open(data_path + args.target).readlines()]
    for pred, tgt in tqdm(zip(output_lines, target_lines), desc='Scoring...'):
        with torch.no_grad():
            output = model(**(tokenizer([pred, tgt], return_tensors='pt', padding=True, max_length=512).to(model.device)))[0]
        tokenized = tokenizer([pred, tgt], return_tensors='pt', padding=True, max_length=512, return_length=True)
        lengths = tokenized['length']
        pred_emb = output[0, :lengths[0]]
        tgt_emb = output[1, :lengths[1]]
        bert1 = cosine_similarity(tgt_emb, pred_emb.unsqueeze(1), dim=-1)
        recall1 = bert1.max(dim=1)[0].mean().data.cpu().numpy()
        precision1 = bert1.max(dim=0)[0].mean().data.cpu().numpy()
        bertf = 2 * (recall1 * precision1) / (recall1 + precision1)
        bert_p.append(precision1)
        bert_r.append(recall1)
        bert_f.append(bertf)
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
    
    print('Rouge-1:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rouge1), np.mean(rouge1p), np.mean(rouge1r)))
    print('Rouge-2:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rouge2), np.mean(rouge2p), np.mean(rouge2r)))
    print('Rouge-L:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(rougel), np.mean(rougelp), np.mean(rougelr)))
    print('BERTScore:\n- F1: {}\n- Precision: {}\n- Recall: {}\n'.format(np.mean(bert_f), np.mean(bert_p), np.mean(bert_r)))


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