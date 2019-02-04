# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import os

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from tensorboardX import SummaryWriter


@register_task('monitored_translation')
class MonitoredTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.num_updates = 0
        should_log = getattr(args, 'distributed_rank', 0) == 0
        log_dir = os.path.join(args.save_dir, 'logs')
        self.logger = SummaryWriter(log_dir) if should_log else None

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        aggregated = super().aggregate_logging_outputs(logging_outputs, criterion)

        if self.logger is not None:
            is_training = criterion.training
            prefix = 'train_' if is_training else 'valid_'
            for key, value in aggregated.items():
                tag = prefix + key
                self.logger.add_scalar(tag, value, self.num_updates)

        return aggregated

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        self.num_updates += 1
        return super().train_step(sample, model, criterion, optimizer, ignore_grad=ignore_grad)
