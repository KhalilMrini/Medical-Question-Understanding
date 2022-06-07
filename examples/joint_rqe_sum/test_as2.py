from fairseq.models.bart import BARTModel
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
import torch
import numpy as np
from rouge_score import rouge_scorer
from nltk import sent_tokenize

for epoch in range(1,2):
    print('For Epoch = ', epoch)
    bart = BARTModel.from_pretrained(
        'checkpoints/',
        checkpoint_file='checkpoint_bestSum_MediqaAS2.pt',#'checkpoint{}Sum_MediqaAS2.pt'.format(epoch),
        data_name_or_path='/vault/datasets/khalil/MediqaAS2-bin'
    )

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    label_fn = lambda label: bart.task.label_dictionary.string(
        [label + bart.task.label_dictionary.nspecial]
    )   
    ncorrect, nsamples = 0, 0
    bart.cuda()
    bart.eval()
    pred_dict = {}
    label_dict = {}
    extractive_summaries = [line.strip() for line in open('/vault/datasets/khalil/MediqaAS2/extractive_summaries.txt').readlines()]
    print('Average number of sentences:', np.mean([len(sent_tokenize(line)) for line in extractive_summaries]))
    extractive_summaries = dict((int(line.split('||')[0]), line.split('||')[1]) for line in extractive_summaries)
    generated_summaries = dict((key, []) for key in extractive_summaries)
    indices = list(extractive_summaries)
    question_idx = 0
    question_to_idx = dict()
    with open('/vault/datasets/khalil/MediqaAS2/dev.tsv') as fin:
        fin.readline()
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            if sent1 not in question_to_idx:
                question_to_idx[sent1] = indices[question_idx]
                question_idx += 1
            tokens = bart.encode(sent1, sent2)
            prediction = bart.predict('sentence_classification_head', tokens[:1024])#.argmax().item()
            pred = torch.nn.functional.softmax(prediction, dim=1).cpu().tolist()
            prediction_label = label_fn(prediction.argmax().item())
            generated_summaries[indices[question_idx - 1]].append([pred[0][1], sent2])
            ncorrect += int(int(prediction_label) == int(target))
            nsamples += 1
            if sent1 not in pred_dict:
                pred_dict[sent1] = []
                label_dict[sent1] = []
            pred_dict[sent1].append(pred[0])
            label_dict[sent1].append([1,0] if int(target) == 0 else [0,1])
    print('| Accuracy: ', float(ncorrect)/float(nsamples))

    rougel = []
    rougelp = []
    rougelr = []
    rouge1 = []
    rouge1p = []
    rouge1r = []
    rouge2 = []
    rouge2p = []
    rouge2r = []
    for idx in indices:
        tgt = extractive_summaries[idx]
        pred = ' '.join([line[-1] for line in sorted([line for line in sorted(generated_summaries[idx], key=lambda x: x[0], reverse=True)[:len(sent_tokenize(tgt))]], key=lambda x: x[1])])#:len(sent_tokenize(tgt))]])
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

    ranks = []
    for query in pred_dict:
        ranked = sorted(list(zip(pred_dict[query], label_dict[query])), key=lambda x: x[0][1], reverse=True)
        idx = 0
        while ranked[idx][1][1] == 0:
            idx += 1
        ranks.append(1/(idx+1))
    
    map_score = np.mean([average_precision_score(label_dict[query], pred_dict[query], average=None)[1] for query in label_dict])
    mrr_score = np.mean(ranks)
    f1score = np.mean([f1_score(label_dict[query], np.round(pred_dict[query]), average=None)[1] for query in label_dict])

    print('| MAP: {}\n| MRR: {}\n| F1: {}'.format(map_score, mrr_score, f1score))