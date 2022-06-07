import math

from fairseq import utils, metrics
from fairseq.data import encoders

from . import FairseqCriterion, register_criterion

from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

import torch
import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
)


@register_criterion('joint_rqe_sum')
class JointRQESumCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.eps = args.label_smoothing
        self.args = args
        self.task = task
        self.debugCount = 0
        args.bpe='gpt2'
        self.bpe = encoders.build_bpe(args)
        self.lambda_factor = args.lambda_factor
        self.classification_head_name = args.classification_head_name
        self.i = 0

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda-factor', default=0.5, type=float, 
                            help='weight between 0.0 and 1.0 to balance the training')
        
        # RQE
        parser.add_argument('--classification-head-name', default='sentence_classification_head',
                            help='name of the classification head to use')
        
        # Joint
        parser.add_argument('--neg-examples', action='store_true', default=False,
                            help='neg_examples switches the loss on the summarization to maximize it when question pairs are not entailed')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # Summarization Output
        sum_net_output = model(**sample['net_input'])

        # RQE Output
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'
        #try:
        if True:
            rqe_logits, _ = model(
                **sample['net_input_rqe'],
                features_only=True,
                classification_head_name=self.classification_head_name,
            )
        #except:
        #    rqe_logits = None

        loss, nll_loss, reward_sum, rqe_loss = self.compute_loss(model, sum_net_output, rqe_logits, sample, reduce=reduce)
        sample_size = sample['target_sum'].size(0) if self.args.sentence_avg else sample['ntokens']
        try:
            nll_loss_item = utils.item(nll_loss.data) if reduce else nll_loss.data
        except:
            nll_loss_item = 0
        try:
            rqe_loss_item = utils.item(rqe_loss.data) if reduce else rqe_loss.data
        except:
            rqe_loss_item = 0
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': nll_loss_item,
            'reward_sum': utils.item(reward_sum) if reduce else reward_sum, # semsim_score : int
            'ntokens': sample['ntokens'],
            'nsentences': sample['target_sum'].size(0),
            'sample_size': sample_size,
            'rqe_loss': rqe_loss_item,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, sum_net_output, rqe_logits, sample, reduce=True):

        debug = False
        self.debugCount += 1
        if self.debugCount % 500 == 1:
            debug = False

        # RQE Loss
        rqe_targets = sample['target_rqe'].view(-1)
        rqe_targets = rqe_targets.float()
        if rqe_logits is not None:
            rqe_loss = F.mse_loss(rqe_logits.view(-1).float(), rqe_targets, reduction='sum')
            rqe_preds = rqe_logits.argmax(dim=1)
        else:
            rqe_loss = 0
        #ncorrect = (rqe_preds == rqe_targets[1::2]).float().mean()

        # Summarization
        lprobs = model.get_normalized_probs(sum_net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        
        sum_target = sample['target_sum']
        flattened_target = sum_target.view(-1, 1)

        # Negative Likelihood Loss
        if flattened_target.dim() == lprobs.dim() - 1:
            flattened_target = flattened_target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=flattened_target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.padding_idx is not None:
            non_pad_mask = flattened_target.ne(self.padding_idx)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        loss = loss / sample['ntokens']
        
        if int(rqe_targets[1]) == 0 and self.args.neg_examples:
            loss = - loss
        
        if debug:
            print("nll_loss, smooth_loss: ",  nll_loss, smooth_loss)
        loss = loss + rqe_loss
        if debug:
            print("==="*10)

        return loss, nll_loss, None, rqe_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2), sample_size, round=3)
        try:
            metrics.log_scalar('nll_loss', sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2), ntokens, round=3)
        except:
            pass
        try:
            metrics.log_scalar('rqe_loss', sum(log.get('rqe_loss', 0) for log in logging_outputs) / sample_size / math.log(2), sample_size, round=3)
        except:
            pass
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('reward_sum', sum(log.get('reward_sum', 0) for log in logging_outputs), 1)
        metrics.log_scalar('sample_loss', sum(log.get('sample_loss', 0) for log in logging_outputs) / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)
        metrics.log_scalar('sample_size', sample_size)
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
