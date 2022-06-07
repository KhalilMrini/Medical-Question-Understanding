from fairseq.models.bart import BARTModel
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
import torch
import numpy as np
from rouge_score import rouge_scorer
from nltk import sent_tokenize
import pickle

MAT_QUESTIONS = {'hcm': """What is the outlook for Periventricular Leukomalacia ?
Is benign familial neonatal seizures inherited ?
What are the treatments for Craniosynostosis ?
Is Pachyonychia congenita inherited ?
How to prevent Parasites - Scabies ?
What is (are) Leukoplakia ?
What is (are) Temporomandibular Joint Dysfunction ?
What are the treatments for Leukoplakia ?
What are the symptoms of Leukoplakia ?
What causes Chronic hiccups ?
Is chronic granulomatous disease inherited ?
What are the treatments for Catamenial pneumothorax ?
What is (are) Emphysema ?
Is Poland syndrome inherited ?
What are the symptoms of COPD ?
What causes Pleurisy and Other Pleural Disorders ?
What causes Chronic hiccups ?
What are the symptoms of Abdominal aortic aneurysm ?
What causes Tietze syndrome ?
What are the treatments for Pneumonia ?
What are the treatments for Tuberculosis (TB) ?
What are the symptoms of Sleep Apnea ?
What is (are) Acute respiratory distress syndrome ?
How to prevent Pericarditis ?
Is idiopathic pulmonary fibrosis inherited ?
What are the treatments for Tuberculosis (TB) ?
Is warfarin resistance inherited ?
What are the treatments for Leukoplakia ?
What is (are) Leukoplakia ?
What are the symptoms of Irritable Bowel Syndrome in Children ?
How to prevent Gum (Periodontal) Disease ?
What are the symptoms of Oral submucous fibrosis ?
What causes Insulin Resistance and Prediabetes ?
What causes Hypoglycemia ?
What are the symptoms of Insulin Resistance and Prediabetes ?
What are the symptoms of Hypoglycemia ?
What causes Chronic hiccups ?
What are the treatments for Pneumonia ?
What are the symptoms of Sarcoidosis ?
How to diagnose Tuberculosis (TB) ?
Is idiopathic pulmonary fibrosis inherited ?
What causes Mixed connective tissue disease ?
Is systemic lupus erythematosus inherited ?
Is androgenetic alopecia inherited ?
What are the treatments for Oral lichen planus ?""".split('\n'),
'meqsum': """What are the symptoms of Anencephaly and spina bifida X-linked ?
What are the symptoms of Congenital rubella ?
What are the symptoms of Nephrogenic diabetes insipidus ?
What are the genetic changes related to Ochoa syndrome ?
What are the genetic changes related to progressive supranuclear palsy ?
What are the genetic changes related to periventricular heterotopia ?
What are the treatments for Lymphatic filariasis ?
Do you have information about HIV/AIDS Medicines
What are the genetic changes related to Klippel-Trenaunay syndrome ?
What is (are) myostatin-related muscle hypertrophy ?
What are the treatments for Tetralogy of Fallot ?
What are the genetic changes related to X-linked adrenoleukodystrophy ?
What are the treatments for Ankylosing spondylitis ?
What are the treatments for Glycogen storage disease type 4 ?""".split('\n')}

REF_QUESTIONS = {'meqsum': [line.split('\t')[1] for line in """0	What are the treatments for spina bifida, vertebral fusion, and syrinx tethered cord?
1	What is the prognosis of rubella in a child?
2	Where can I find information on diabetes insipidus?
4	Where can I find information on ochoa syndrome?
5	What are the treatments for progressive supranuclear palsy and how can I find physician(s) who specialize in it?
6	Where can I find information on periventricular heterotopia, including the latest research?
7	What are the treatments for filarial disease?
18	What are the treatments for HIV/AIDS?
19	What is the latest research on klippel-trenaunay syndrome, and what are the treatments for it?
21	Where can I find information on myostatin-related muscular hypertrophy, and how can I find an organization doing research on it?
32	Which organizations provide support for Tetralogy of Fallot in Peru?
36	Where can I find information on adrenoleukodystrophy?
43	What are the treatments for Ankylosing Spondylitis?
47	What are the treatments for Glycogen storage disease?""".split('\n')],
'hcm': [line.split('\t')[1] for line in """2	What is meant by periventricular leukomalacia?
6	What causes neonatal seizures?
7	How to treat Craniosynostosis?
9	Is paronychia infection common in babies?
10	What causes scabies?
18	Does leukoplakia cause adverse side effects?
19	How can temporomandibular joint dysfunction be treated?
20	How can leukoplakia be treated?
23	How to test if leukoplakia is cancerous?
26	What causes persistent hiccups?
27	What is granulomatous disease?
33	What is the treatment for pneumothorax?
36	How can emphysema causing breathlessness be treated?
39	Is Poland syndrome genetic in nature?
41	What are the symptoms of COPD?
42	What causes Pleurisy?
45	What causes recurrent hiccups?
57	What are the symptoms of aortic aneurysm?
58	What causes tietze syndrome?
60	What are the symptoms and treatment for Pneumonia?
63	What is the treatment for Tuberculosis and lymph nodes?
65	What are apnea symptoms?
66	What causes respiratory distress ?
67	How to treat recurrent pericarditis?
74	What could cause pulmonary fibrosis?
75	What are the symptoms of Pulmonary Tuberculosis?
80	Can a body rash be due to warfarin?
83	How is leukoplakia treated?
86	What causes leukoplakia on tongue?
88	What are the symptoms and treatment for Irritable Bowel Syndrome?
101	How to prevent periodontitis gum disease?
103	Can Oral sub-mucous fibrosis be recovered by surgery ?
107	What causes prediabetes?
110	What causes hypoglycemia while suffering from diabetes?
113	What are prediabetes symptoms?
115	What are the symptoms and treatment for hypoglycemia?
120	What would cause random hiccups?
124	What is the treatment procedure for Pneumonia tuberculosis?
129	What are the symptoms of Sarcoidosis?
132	How to confirm Tuberculosis?
133	What is idiopathic pulmonary fibrosis?
141	What is mixed connective tissue disease?
150	What is Systemic lupus erythometosus?
160	Are recommended medications for alopecia effective?
164	Which is the effective therapy for lichen planus ?""".split('\n')]}

for dataset in ['hcm', 'meqsum']:
    print('For dataset = ', dataset)
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
    database = pickle.load(open('/vault/datasets/khalil/EndToEndQA/{}_database.pkl'.format(dataset), 'rb'))
    questions_as_str = pickle.load(open('/vault/datasets/khalil/EndToEndQA/{}_questions_as_str.pkl'.format(dataset), 'rb'))
    ref_output = open('pred_ref_{}.tsv'.format(dataset), 'w')
    mat_output = open('pred_mat_{}.tsv'.format(dataset), 'w')
    assert len(REF_QUESTIONS[dataset]) == len(MAT_QUESTIONS[dataset])
    for q_idx in range(len(MAT_QUESTIONS[dataset])):
        sent1 = MAT_QUESTIONS[dataset][q_idx]
        idx = questions_as_str.index(sent1)
        ref_question = REF_QUESTIONS[dataset][q_idx]
        print(dataset, q_idx, sent1)
        pred_dict = []
        ref_pred_dict = []
        all_answers = [answer.strip() for answer in database[idx] if answer.strip()[-1] not in ['?', ')', ']'] and 
                    'The Human Phenotype Ontology provides' not in answer 
                    and 'Much of this information comes from Orphanet' not in answer 
                    and 'NIH' not in answer and 'In these cases, the sign or symptom may be rare or common.' not in answer 
                    and not answer.startswith('Get more details') and not answer.startswith('See risk factors for') 
                    and 'this page' not in answer and 'this website' not in answer 
                    and 'Centers for Disease Control and Prevention' not in answer and not answer.startswith('For information on') 
                    and 'If the information is available, the table below includes' not in answer 
                    and 'You can use the MedlinePlus Medical Dictionary' not in answer
                    and 'For more information, please see the Exit Notification and Disclaimer policy.' not in answer
                    and 'This graphic notice means that you are leaving an HHS Web site' not in answer
                    and not answer.startswith('These resources address the diagnosis or management of')
                    and "of Health and Human Services Office on Women's Health" not in answer]
        for a_idx in range(len(all_answers)):
            sent2 = all_answers[a_idx].replace('\n', '').replace('\t', '')
            tokens = bart.encode(sent1, sent2)
            prediction = bart.predict('sentence_classification_head', tokens[:1024])#.argmax().item()
            pred = torch.nn.functional.softmax(prediction, dim=1).cpu().tolist()
            pred_dict.append(pred[0])
            tokens = bart.encode(ref_question, sent2)
            prediction = bart.predict('sentence_classification_head', tokens[:1024])#.argmax().item()
            pred = torch.nn.functional.softmax(prediction, dim=1).cpu().tolist()
            ref_pred_dict.append(pred[0])
        ranked = sorted(list(zip(pred_dict, all_answers)), key=lambda x: x[0][1], reverse=True)
        ref_ranked = sorted(list(zip(ref_pred_dict, all_answers)), key=lambda x: x[0][1], reverse=True)
        for rank in ranked:
            print('{}\t{}\t{}\t{}'.format(q_idx, sent1, rank[1].replace('\n', ' ').replace('\t', ' '), rank[0][1]), file=mat_output)
        for rank in ref_ranked:
            print('{}\t{}\t{}\t{}'.format(q_idx, sent1, rank[1].replace('\n', ' ').replace('\t', ' '), rank[0][1]), file=ref_output)
    ref_output.close()
    mat_output.close()
