"""Callbacks that do some kind of model saving stuff."""
from lightning.pytorch.callbacks import ModelCheckpoint


class BestCheckpoint(ModelCheckpoint):
    """Wrapper around ModelCheckpoint that changes the "last" checkpoint name.
    
    `ModelCheckpoint` supports a feature where it will create a copy of the last checkpoint saved,
    and call it "last". This one simply renames that to "best". The idea is to have two checkpoint
    callbacks: One simply saves the latest checkpoints, the other saves only the best by some
    metric. By renaming "last" here, we can use `save_last` for both callbacks, while avoiding
    naming confusion.
    """
    CHECKPOINT_NAME_LAST = "best"
