import os
import yaml
from addict import Dict
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import get_dataset
from pl_modules import get_pl_module
from normalize_data import SCALERS_DICT


def train(data_config, train_config):
    exp_output_dir = os.path.join(train_config.output_dir, train_config.exp_name)
    checkpoint_dir = os.path.join(exp_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=exp_output_dir,
        name="logs",
        default_hp_metric=False,
    )

    callbacks = []

    checkpoint_best_loss = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_loss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    checkpoint_best_mse = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_mse-{epoch:02d}-{val_metric_mse:.4f}",
        save_top_k=5,
        monitor="val_metric_mse",
        mode="min",
    )

    checkpoint_periodic = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=5,
        save_top_k=-1,
    )

    callbacks.append(checkpoint_best_loss)
    if train_config.dataset.val_metric:
        callbacks.append(checkpoint_best_mse)
    callbacks.append(checkpoint_periodic)

    train_dataset = get_dataset(
        dataset_type=train_config.dataset.type,
        data_path=os.path.join(
            data_config.input_data.root_data_folder, 
            data_config.preprocess_params.exp_data_folder, 
            "train_dataset.pkl"
        ),
        scalers_dict=SCALERS_DICT,
        target_columns=train_config.dataset.target_columns,
        grid_size=data_config.preprocess_params.grid_size,
        mode="train",
    )
    val_dataset = get_dataset(
        dataset_type=train_config.dataset.type,
        data_path=os.path.join(
            data_config.input_data.root_data_folder, 
            data_config.preprocess_params.exp_data_folder, 
            "val_dataset.pkl"
        ),
        scalers_dict=SCALERS_DICT,
        target_columns=train_config.dataset.target_columns,
        grid_size=data_config.preprocess_params.grid_size,
        mode="val",
    )

    item = train_dataset[0]
    history_len, future_len = len(item["history"]), len(item["future"])
    print(item["history"].shape)
    print(item["y"].shape)
    print(item["future"].shape)
    print(item["static"].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=train_config.dataset.train_bs, num_workers=8, shuffle=True,)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config.dataset.val_bs, num_workers=8, shuffle=False,)

    pl_model = get_pl_module(train_config, scalers_dict=SCALERS_DICT)

    trainer = pl.Trainer(
        max_epochs=train_config.train_params.n_epochs,
        accelerator=train_config.train_params.device,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=32
    )

    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--train_config', type=str, required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Dict(config)


if __name__ == "__main__":
    args = parse_args()
    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)

    print(train_config.dataset.target_columns)

    train(data_config, train_config)
