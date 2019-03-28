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


@register_task('monitored_translation')
class MonitoredTranslationTask(TranslationTask):
    """
    Provides tensorboard logging to FairseqTask's .
    Note that for validation outputs we potentially get the losses
    of the several batches in the validation data, so we aggregate
    them until the next batch of training data arrives.
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        should_log = getattr(args, 'distributed_rank', 0) == 0
        if should_log:
            train_logger = SummaryWriter(os.path.join(args.save_dir, 'train_logs'))
            valid_logger = SummaryWriter(os.path.join(args.save_dir, 'valid_logs'))
            for logger in [train_logger, valid_logger]:
                logger.add_text('args', str(vars(args)))
                logger.add_text('sys.argv', " ".join(sys.argv))
        else:
            train_logger = valid_logger = None

        self.should_log = should_log
        self.train_logger = train_logger
        self.valid_logger = valid_logger
        self.num_updates = 0
        self.last_validation_outputs = None
        self.last_validation_step = 0

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        aggregated = super().aggregate_logging_outputs(logging_outputs, criterion)

        if not self.should_log:
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
        logger = self.train_logger if is_training else self.valid_logger
        for key, value in aggregated.items():
            logger.add_scalar(key, value, step)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        self.num_updates += 1
        return super().train_step(sample, model, criterion, optimizer, ignore_grad=ignore_grad)


