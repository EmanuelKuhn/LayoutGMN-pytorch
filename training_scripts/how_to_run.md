# How to run


Run with:

```!python train_TRI.py --train_mode --cuda```


# Differences between reproduction and reference implementation

- APN dict: Uses a custom one also based on IoU
- negative sampling: the custom apn dict doesn't contain low IoUs, thus negatives are sampled from the remaining training examples (like in the original GCN-CNN repository)