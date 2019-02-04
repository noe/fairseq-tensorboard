# fairseq-tensorboard

This is a small utility to monitor fairseq training in tensorboard.

It is not a fork of fairseq, but just a small class that extends its
functionality with tensorboard logging.

# Installation and Usage

You just need to clone fairseq-tensorboard, install its only dependency
([tensorboardX](https://github.com/lanpa/tensorboardX)) and launch
fairseq's `train.py` specifying task `monitored_translation`:

```
pip install tensorboardX

git clone https://github.com/noe/fairseq-tensorboard.git

python fairseq/train.py \
   --user-dir ./fairseq-tensorboard/fstb \
   --task monitored_translation [...]
```
