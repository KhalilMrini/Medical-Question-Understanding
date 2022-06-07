from fairseq.models.bart import BARTModel
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
import torch
import numpy as np
from rouge_score import rouge_scorer
from nltk import sent_tokenize

bart = BARTModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_bestMediqaAS2_Finetuned.pt',
    data_name_or_path='/vault/datasets/khalil/MediqaAS2-bin'
)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)
bart.cuda()
bart.eval()
extractive_summaries = [line.strip() for line in open('/vault/datasets/khalil/MediqaAS2/test_questions.txt').readlines()]
extractive_summaries = dict((int(line.split('||')[0]), line.split('||')[1]) for line in extractive_summaries)
generated_summaries = dict((key, []) for key in extractive_summaries)
indices = list(extractive_summaries)
question_idx = 0
question_to_idx = dict()
with open('/vault/datasets/khalil/MediqaAS2/test.tsv') as fin:
    fin.readline()
    for index, line in tqdm(enumerate(fin)):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[1], tokens[2]
        if sent1 not in question_to_idx:
            question_to_idx[sent1] = indices[question_idx]
            question_idx += 1
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens[:1024])#.argmax().item()
        pred = torch.nn.functional.softmax(prediction, dim=1).cpu().tolist()
        prediction_label = label_fn(prediction.argmax().item())
        generated_summaries[indices[question_idx - 1]].append([pred[0][1], index, sent2])

output_file = open('output_as2.txt', 'w')
for idx in indices:
    tgt = extractive_summaries[idx]
    pred = ' '.join([line[-1] for line in sorted([line for line in sorted(generated_summaries[idx], key=lambda x: x[0], reverse=True)[:11]], key=lambda x: x[1])])
    print('{}\t{}'.format(idx, pred), file=output_file)

output_file.close()