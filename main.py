from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler
import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np

# for parallelization
import lightning.pytorch as li
from lightning.pytorch.strategies import DDPStrategy

# local imports
from vit3D import ViT
from dataset import RNSA2023Dataset
from dataloader import collate_fn_pad
from litvit import LitViT

# logs
import tensorboard
from lightning.pytorch.loggers import WandbLogger

BASE_PATH = Path("/home/infres/jma-21/J2Letters_RNSA_2023/rsna-2023-abdominal-trauma-detection")
PT_PATH = Path("/home/infres/jma-21/J2Letters_RNSA_2023/rnsa-2023-5mm-slices-pt")
OUTPUT_PATH = Path("/home/infres/jma-21/J2Letters_RNSA_2023/output")

LABEL_COLS = ["bowel_healthy", "bowel_injury", "extravasation_healthy", "extravasation_injury", "kidney_healthy", "kidney_low", "kidney_high", "liver_healthy", "liver_low", "liver_high", "spleen_healthy", "spleen_low", "spleen_high", "any_injury"]
FRAME_PATCH_SIZE = 8
BATCH_SIZE = 2
FRAMES = 512 # 303 is the max in the train data 

wandb_logger = WandbLogger(project="RNSA_2023_ViT3D")

# Instantiate model
model = ViT(
    image_size = 128,          # image size
    frames = FRAMES,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = FRAME_PATCH_SIZE,      # frame patch size
    dim = 512,
    depth = 6,
    heads = 4,
    mlp_dim = 1024,
    neck_dim = 32,
    channels = 1,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Data prep
train_labels = pl.read_csv(BASE_PATH.joinpath("train.csv"))
train_patient_series = pl.read_csv(BASE_PATH.joinpath("train_series_meta.csv"))

counts_unique_labels = train_labels[LABEL_COLS].group_by(LABEL_COLS).count().with_row_count(offset=1)
counts_unique_labels = counts_unique_labels.with_columns(pl.when(pl.col("count") == 1).then(0).otherwise(pl.col("row_nr")).alias("group"))
# Add weights to each group. Later used in WeightedRandomSampler
# 2292 corresponds to the largest group by far, with healthy people inside
counts_unique_labels = counts_unique_labels.with_columns(pl.when(pl.col("count") == 2292).then((len(counts_unique_labels)-1)/pl.col("count")).otherwise(1/pl.col("count")).alias("weight"))
train_labels_group = train_labels.join(counts_unique_labels.select(LABEL_COLS + ["group", "weight"]), on=LABEL_COLS, how="left")

train_idx, valid_idx= train_test_split(np.arange(len(train_labels)), test_size=0.1, shuffle=True, stratify=train_labels_group["group"])

# Create datasets
train_dataset = RNSA2023Dataset(train_labels_group, train_idx)
valid_dataset = RNSA2023Dataset(train_labels_group, valid_idx)

# Samplers
train_samples_weight = train_dataset.labels.select(pl.col("weight")).to_list()
train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))
valid_samples_weight = valid_dataset.labels.select(pl.col("weight")).to_list()
valid_sampler = WeightedRandomSampler(valid_samples_weight, len(valid_samples_weight))

# Create data loaders for our datasets; shuffle for training, not for validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn_pad, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = collate_fn_pad, sampler=valid_sampler)

# Init the LitVit
lit_vit = LitViT(model)

# log gradients, parameter histogram and model topology
wandb_logger.watch(lit_vit, log="all")

# Explicitly specify the process group backend if you choose to
ddp = DDPStrategy(process_group_backend="gloo")

# Init trainer
trainer = li.Trainer(devices="auto", 
                     accelerator="gpu", 
                     strategy=ddp,
                     max_epochs=100, #limit_train_batches=0.25,
                     log_every_n_steps=50,
                     logger=wandb_logger)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer.fit(model=lit_vit, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# Done
print("-------- TRAINING DONE --------")
