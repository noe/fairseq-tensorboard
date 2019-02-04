# fairseq-tensorboard

This is a small utility to monitor fairseq training in tensorboard.

It is not a fork of fairseq, but just a small class that extends its
functionality with tensorboard logging.

# Installation and Usage

You just need to clone fairseq-tensorboard, install its only direct dependency
apart from fairseq itself
([tensorboardX](https://github.com/lanpa/tensorboardX)) and launch
fairseq's `train.py` specifying as task `monitored_translation`:

```
pip install tensorboardX

git clone https://github.com/noe/fairseq-tensorboard.git

python fairseq/train.py \
   --user-dir ./fairseq-tensorboard/fstb \
   --task monitored_translation [...]
```

# FAQ

### How can fairseq load fstb?

You have to provide fairseq with command line argument `--user-dir` with
the path of fstb. This instructs fairseq to load the fstb code, which
registers task `monitored_translation`.

### Does fstb work with multi-GPU training?

Yes, it has been tested with single-node multi-GPU training. Only the
first worker process logs to tensorboard. The behaviour of the
remaining workers is unaltered.
