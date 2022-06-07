from fairseq.models.bart import BARTModel
from tqdm import tqdm

for lambda_ in range(11):
    this_lambda = str(float(lambda_/10.0))
    print('For Lambda = ', this_lambda)
    bart = BARTModel.from_pretrained(
        'checkpoints/',
        checkpoint_file='checkpoint_bestRQE_Joint_{}.pt'.format(this_lambda),
        data_name_or_path='/vault/datasets/khalil/RQE-bin'
    )

    label_fn = lambda label: bart.task.label_dictionary.string(
        [label + bart.task.label_dictionary.nspecial]
    )   
    ncorrect, nsamples = 0, 0
    bart.cuda()
    bart.eval()
    with open('/vault/datasets/khalil/RQE/dev.tsv') as fin:
        fin.readline()
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens = bart.encode(sent1, sent2)
            prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))