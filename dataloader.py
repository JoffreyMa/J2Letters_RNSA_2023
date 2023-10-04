from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
import math

FRAME_PATCH_SIZE = 4
FRAMES = 512

# Adapted from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/18
def collate_fn_pad(items):
    data_seq = []
    label_seq = []
    max_frame = 0 
    for data, label in items:
        if data.shape[0] > max_frame:
            max_frame = data.shape[0]
        data_seq.append(data)
        label_seq.append(label)
    #big_one = torch.ones((FRAME_PATCH_SIZE*(math.floor(max_frame/FRAME_PATCH_SIZE)+1), data.shape[1], data.shape[2])) # largest size dividable by FRAME_PATCH_SIZE
    big_one = torch.ones((FRAMES, data.shape[1], data.shape[2]))
    data_seq.append(big_one)
    data_seq_batched = pad_sequence(data_seq, batch_first=True).unsqueeze(1)[:-1] # unsqueeze to add channel, do not keep the big one
    label_seq_batched = torch.cat(label_seq)
    assert data_seq_batched.shape[0] == len(label_seq_batched)
    return data_seq_batched.to(torch.float32), label_seq_batched.to(torch.float32)
