# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    ConcatSentencesDataset,
    data_utils,
    encoders,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    JointSumDataset
)

from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


@register_task('joint_rqe_sum')
class JointRQESumTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')

        # RQE
        parser.add_argument('--num-classes', type=int, default=2,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='train,valid',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='add prev_output_tokens to sample, used for encoder-decoder arch')

        # Summarization
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the sequence')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # Low-Resource Setting
        parser.add_argument('--max-datapoints', default=0, type=int, metavar='N',
                            help='if >0, then we only train with a limited number of datapoints')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

        # generation args
        parser.add_argument('--beam', default=4, type=int)
        parser.add_argument('--lenpen', default=2.0, type=float)
        parser.add_argument('--min-len', default=10, type=int)
        parser.add_argument('--max-len-a', default=0, type=int)
        parser.add_argument('--max-len-b', default=20, type=int)
        parser.add_argument('--no-repeat-ngram-size', default=3, type=int)
        

    def __init__(self, args, src_dict, tgt_dict, rqe_dict, label_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.rqe_dict = rqe_dict
        self.label_dict = label_dict
        

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        rqe_dict = cls.load_dictionary(os.path.join(paths[0], 'input0', 'dict.txt'))
        label_dict = cls.load_dictionary(os.path.join(paths[0], 'label', 'dict.txt'))
        logger.info('[source] dictionary: {} types'.format(len(src_dict)))
        logger.info('[target] dictionary: {} types'.format(len(tgt_dict)))
        logger.info('[label] dictionary: {} types'.format(len(label_dict)))

        return cls(args, src_dict, tgt_dict, rqe_dict, label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        ## RQE Dataset
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset('input0', self.rqe_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path(type, split))
        input1 = make_dataset('input1', self.rqe_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions,
            self.args.seed,
        )

        rqe_input = src_tokens#RightPadDataset(src_tokens,
                    #pad_idx=self.source_dictionary.pad())

        label_dataset = make_dataset('label', self.label_dictionary)
        if label_dataset is not None:
            label_dataset = OffsetTokensDataset(
                StripTokenDataset(label_dataset,
                    id_to_strip=self.label_dictionary.eos()),
                offset=-self.label_dictionary.nspecial)

        ## Summarization Dataset
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = self.load_sum_dataset(
            data_path, split, rqe_input, label_dataset, 
            combine=combine, shuffle=(split != 'test'))

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return JointSumDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        
        class_head = getattr(args, 'classification_head_name', 'sentence_classification_head')
        logger.info('Current classification heads', list(model.classification_heads))
        if class_head not in model.classification_heads:
            model.register_classification_head(
                class_head,
                num_classes=self.args.num_classes,
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def rqe_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.rqe_dict

    @property
    def label_dictionary(self):
        return self.label_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def load_sum_dataset(self, data_path, split, rqe_input, label_dataset, 
        combine, shuffle=True, append_source_id=False, prepend_bos=False):

        src, src_dict = self.args.source_lang, self.src_dict
        tgt, tgt_dict = self.args.target_lang, self.tgt_dict
        dataset_impl=self.args.dataset_impl
        upsample_primary=self.args.upsample_primary
        left_pad_source=self.args.left_pad_source
        left_pad_target=self.args.left_pad_target
        max_source_positions=self.args.max_source_positions
        max_target_positions=self.args.max_target_positions
        load_alignments=self.args.load_alignments
        truncate_source=self.args.truncate_source
        num_buckets=self.args.num_batch_buckets

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src, data_path):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

            logger.info('{} {} {}-{} {} examples'.format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            ))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

        eos = None
        if append_source_id:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
            eos = tgt_dict.index('[{}]'.format(tgt))

        align_dataset = None
        if load_alignments:
            align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
            if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
        return JointSumDataset(
            src_dataset, src_dataset.sizes, src_dict, rqe_input,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            label_dataset,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            add_prev_output_tokens=self.args.add_prev_output_tokens,
            source_pad=self.rqe_dictionary.pad(),
            max_positions=self.args.max_positions
        )
