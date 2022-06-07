# Medical Question Understanding

## 1. Papers included in this repository

<li><b><i>"A Gradually Soft Multi-Task and Data-Augmented Approach to Medical Question Understanding"</i></b> <img height="16" src="https://khalilmrini.github.io/images/nih.png" width="24" style="display: inline-block;"/> <img height="16" src="https://khalilmrini.github.io/images/adobe.png" width="16" style="display: inline-block;"/><br>
<b>Khalil Mrini</b>, Franck Dernoncourt, Seunghyun Yoon, Trung Bui, Walter Chang, Emilia Farcas, Ndapa Nakashole<br>
ACL 2021 (Main, Long Paper)<br>
<a href="https://aclanthology.org/2021.acl-long.119.pdf">PDF</a> | <a href="https://aclanthology.org/2021.acl-long.119/">ACL Anthology</a> | <a href="https://aclanthology.org/2021.acl-long.119.bib">BibTeX</a> | <a href="https://www.youtube.com/watch?v=TOBXl6d3uLI&ab_channel=KhalilMrini">Video Presentation</a></li>

<li><b><i>"Joint Summarization-Entailment Optimization for Consumer Health Question Understanding"</i></b> <img height="16" src="https://khalilmrini.github.io/images/nih.png" width="24" style="display: inline-block;"/> <img height="16" src="https://khalilmrini.github.io/images/adobe.png" width="16" style="display: inline-block;"/><br>
<b>Khalil Mrini</b>, Franck Dernoncourt, Walter Chang, Emilia Farcas, Ndapa Nakashole<br>
NAACL 2021 workshop on NLP for Medical Conversations (NLPMC)<br>
<b>Best Student Paper Award</b><br>
<a href="https://www.aclweb.org/anthology/2021.nlpmc-1.8.pdf">PDF</a> | <a href="https://www.aclweb.org/anthology/2021.nlpmc-1.8/">ACL Anthology</a> | <a href="https://www.aclweb.org/anthology/2021.nlpmc-1.8.bib">BibTeX</a></li>

<li><b><i>"UCSD-Adobe at MEDIQA 2021: Transfer Learning and Answer Sentence Selection for Medical Summarization"</i></b> <img height="16" src="https://khalilmrini.github.io/images/nih.png" width="24" style="display: inline-block;"/> <img height="16" src="https://khalilmrini.github.io/images/adobe.png" width="16" style="display: inline-block;"/><br>
<b>Khalil Mrini</b>, Franck Dernoncourt, Seunghyun Yoon, Trung Bui, Walter Chang, Emilia Farcas, Ndapa Nakashole<br>
NAACL 2021 workshop on Biomedical NLP (BioNLP)<br>
<a href="https://www.aclweb.org/anthology/2021.bionlp-1.28.pdf">PDF</a> | <a href="https://www.aclweb.org/anthology/2021.bionlp-1.28">ACL Anthology</a> | <a href="https://www.aclweb.org/anthology/2021.bionlp-1.28.bib">BibTeX</a></li>

## 2. Installation

Use the following commands to install this package:

```
pip install --editable ./
pip install transformers
```

Download pre-trained models like this:

```
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz
tar -xzvf bart.large.xsum.tar.gz
rm bart.large.xsum.tar.gz
cd ..
```

## 3. Data preprocessing

Preprocess the MeQSum dataset as follows, where "MeQSum" is the same folder as the one (parent folder) where this repository is located:

```
sh ./examples/joint_rqe_sum/preprocess_MeQSum.sh .. MeQSum
mv MeQSum-bin ../MeQSum-bin

TASK=../MeQSum
for SPLIT in train dev
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/dev.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

## 4. Training Commands

To train the model with the Gradually Soft Parameter Sharing loss on MeQSum:

```
TOTAL_NUM_UPDATES=810 
WARMUP_UPDATES=81
LR=3e-05
MAX_TOKENS=512
UPDATE_FREQ=4
BART_PATH=./models/bart.large.xsum/model.pt

CUDA_VISIBLE_DEVICES=1 fairseq-train ../MeQSum-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task joint_rqe_sum \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --batch-size 8 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large_gradsoft \
    --criterion grad_soft --add-prev-output-tokens \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

To train the model with the Joint Summarization-Entailment loss on MeQSum:

```
TOTAL_NUM_UPDATES=810 
WARMUP_UPDATES=81
LR=3e-05
MAX_TOKENS=512
UPDATE_FREQ=4
BART_PATH=./models/bart.large.xsum/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train ../MeQSum-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task joint_rqe_sum \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --beam 2 \
    --batch-size 8 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion joint_rqe_sum --add-prev-output-tokens \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --no-epoch-checkpoints --no-last-checkpoints \
    --clip-norm 0.1 \
    --max-epoch 10 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

To train the model with the Summarization loss on MeQSum:

```
TOTAL_NUM_UPDATES=5400 
WARMUP_UPDATES=81
LR=3e-05
MAX_TOKENS=512
UPDATE_FREQ=4
BART_PATH=./models/bart.large.xsum/model.pt

CUDA_VISIBLE_DEVICES=1 fairseq-train ../MeQSum-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --batch-size 8 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-epoch 100 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test --no-epoch-checkpoints --no-last-checkpoints \
    --find-unused-parameters;
```

## Fairseq License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

## Fairseq Citation

Please cite as:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
