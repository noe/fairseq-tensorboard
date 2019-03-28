# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import os
import sys

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from tensorboardX import SummaryWriter


class LoggingMixin:
    """
    Mixin that provides tensorboard logging to FairseqTask's .
    Note that for validation outputs we potentially get the losses
    of the several batches in the validation data, so we aggregate
    them until the next batch of training data arrives.
    """

    def __init__(self, args):
        should_log = getattr(args, 'distributed_rank', 0) == 0
        if should_log:
            log_dir = os.path.join(args.save_dir, 'logs')
            logger = SummaryWriter(log_dir)
            logger.add_text('args', str(vars(args)))
            logger.add_text('sys.argv', " ".join(sys.argv))
        else:
            logger = None

        self.logger = logger
        self.num_updates = 0
        self.last_validation_outputs = None
        self.last_validation_step = 0

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        aggregated = super().aggregate_logging_outputs(logging_outputs, criterion)

        if self.logger is None:
            return aggregated

        is_training = criterion.training

        if not is_training:
            # collect outputs for later

            if self.last_validation_outputs is None:
                self.last_validation_outputs = dict(aggregated)
                self.last_validation_step = self.num_updates
            else:
                # aggregate new outputs with the last received ones
                last_ntokens = self.last_validation_outputs['ntokens']
                new_ntokens = aggregated['ntokens']
                last_outputs_weight = last_ntokens / float(last_ntokens + new_ntokens)
                new_outputs_weight = new_ntokens / float(last_ntokens + new_ntokens)

                for key in aggregated.keys():
                    if "loss" in key:
                        new_loss = (last_outputs_weight * self.last_validation_outputs[key]
                                    + new_outputs_weight * aggregated[key])
                        self.last_validation_outputs[key] = new_loss
                    else:
                        self.last_validation_outputs[key] += aggregated[key]
        else:
            if self.last_validation_outputs is not None:
                self._log_outputs(self.last_validation_outputs,
                                  self.last_validation_step,
                                  False)
                self.last_validation_outputs = None
            self._log_outputs(aggregated, self.num_updates, True)

        return aggregated

    def _log_outputs(self, aggregated, step, is_training):
        prefix = 'train_' if is_training else 'valid_'
        for key, value in aggregated.items():
            tag = prefix + key
            self.logger.add_scalar(tag, value, step)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        self.num_updates += 1
        return super().train_step(sample, model, criterion, optimizer, ignore_grad=ignore_grad)


@register_task('monitored_translation')
class MonitoredTranslationTask(TranslationTask, LoggingMixin):
    def __init__(self, args, src_dict, tgt_dict):
        super(TranslationTask, self).__init__(args, src_dict, tgt_dict)
        super(LoggingMixin, self).__init__(args)
