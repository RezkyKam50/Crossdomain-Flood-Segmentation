import os
import sys
import numpy as np
import torch

import terratorch
from terratorch.datamodules import Landslide4SenseNonGeoDataModule
from terratorch.datasets import Landslide4SenseNonGeo
from terratorch.tasks import SemanticSegmentationTask

import albumentations

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

DATASET_PATH = "./datasets"
OUT_DIR = "./landslide_example"  # where to save checkpoints and log files

BATCH_SIZE = 16
EPOCHS = 40
LR = 1.0e-4
WEIGHT_DECAY = 0.1
HEAD_DROPOUT=0.1
FREEZE_BACKBONE = False
BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
NUM_WORKERS = 7   # adjust value based on your system
SEED = 0


def load():
    data_module = Landslide4SenseNonGeoDataModule(
        data_root=DATASET_PATH,
    )
    print(data_module.means)
    print(data_module.stds)

    data_module.setup("fit")
    train_dataset = data_module.train_dataset
    len(train_dataset)

    print(train_dataset.all_band_names)

    for i in range(5):
        train_dataset.plot(train_dataset[i])


    val_dataset = data_module.val_dataset
    print(len(val_dataset))

    data_module.setup("test")
    test_dataset = data_module.test_dataset
    print(len(test_dataset))
    pl.seed_everything(SEED)


def setup():
    # Logger
    logger = TensorBoardLogger(
        save_dir=OUT_DIR,
        name="landslide_example",
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        dirpath=os.path.join(OUT_DIR, "landslide_example", "checkpoints"),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        precision="bf16-mixed",
        num_nodes=1,
        logger=logger,
        max_epochs=EPOCHS,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    return trainer, checkpoint_callback

def preproc():
    # DataModule
    transforms = [
        albumentations.Resize(224, 224),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]

    # Adding augmentations for training
    train_transforms = [
        albumentations.HorizontalFlip(),
        albumentations.Resize(224, 224),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]

    return transforms, train_transforms

def train():

    transforms, train_transforms = preproc()
    trainer, checkpoint_callback = setup()

    data_module = Landslide4SenseNonGeoDataModule(
        batch_size=BATCH_SIZE,
        bands=BANDS,
        data_root=DATASET_PATH,
        train_transform=train_transforms,
        val_transforms=transforms,
        test_transforms=transforms,
        num_workers=NUM_WORKERS,
    )


    backbone_args = dict(
        backbone_pretrained=True,
        backbone="prithvi_eo_v2_300", # prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
        backbone_bands=BANDS,
        backbone_num_frames=1,
    )

    decoder_args = dict(
        decoder="UperNetDecoder",
        decoder_channels=256,
        decoder_scale_modules=True,
    )

    necks = [
        dict(
                name="SelectIndices",
                # indices=[2, 5, 8, 11]    # indices for prithvi_eo_v1_100
                indices=[5, 11, 17, 23],   # indices for prithvi_eo_v2_300
                # indices=[7, 15, 23, 31]  # indices for prithvi_eo_v2_600
            ),
        dict(
                name="ReshapeTokensToImage",
            )
        ]

    model_args = dict(
        **backbone_args,
        **decoder_args,
        num_classes=2,
        head_dropout=HEAD_DROPOUT,
        head_channel_list=[128, 64],
        necks=necks,
        rescale=True,
    )
        

    model = SemanticSegmentationTask(
        model_args=model_args,
        plot_on_val=False,
        loss="focal",
        lr=LR,
        optimizer="AdamW",
        scheduler="StepLR",
        scheduler_hparams={"step_size": 10, "gamma": 0.9},
        optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
        ignore_index=-1,
        freeze_backbone=FREEZE_BACKBONE,
        freeze_decoder=False,
        model_factory="EncoderDecoderFactory",
    )
    # Training
    trainer.fit(model, datamodule=data_module)
    ckpt_path = checkpoint_callback.best_model_path

    # Test results
    test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    print("Test Results:", test_results)


if __name__ == "__main__":
    train()





