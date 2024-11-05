import os
import yaml
from addict import Dict
import argparse
import joblib
import pickle
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from data_utils import read_testcase_ids, read_testcases, get_validation_split
from data_utils import interpolate_case_to_grid, interpolate_case_to_grid_test
from data_utils import get_speed_stats, get_scale_stats
from data_utils import add_speed_to_data, add_sin_cos_yaw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--only_test', type=bool, required=False, default=False)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = Dict(config)

    return config

def prepare_val_split(config, train_folder, val_split_path):
    if os.path.exists(val_split_path):
        logging.info(f"Loading validation split from {val_split_path}")
        split_info = joblib.load(val_split_path)
        train_ids, val_ids = split_info["train_ids"], split_info["val_ids"]
    else:
        full_train_ids = read_testcase_ids(train_folder)
        logging.info(f"Full train ids: {len(full_train_ids)}")

        val_size = config.preprocess_params.val_size
        train_ids, val_ids = get_validation_split(full_train_ids, val_size)
        logging.info(f"split train ids len: {len(train_ids)}, split val ids len: {len(val_ids)}")
        joblib.dump({"train_ids": train_ids, "val_ids": val_ids}, val_split_path)


def preprocess_data(config, input_folder, processed_folder, val_split_path, mode):
    if mode == "test":
        dataset = read_testcases(input_folder, is_test=True)
    else:
        ids = joblib.load(val_split_path)[mode + "_ids"]
        dataset = read_testcases(input_folder, testcase_ids=ids)

    logging.info("[preprocess data][interpolate]")
    grid_size = config.preprocess_params.grid_size
    if mode == "test":
        dataset = {k: interpolate_case_to_grid_test(v, grid_size) for k,v in tqdm(dataset.items())}
    else:
        dataset = {k: interpolate_case_to_grid(v, grid_size) for k,v in tqdm(dataset.items())}

    logging.info("[preprocess data][add speed]")
    dataset = add_speed_to_data(dataset, grid_size)

    # logging.info("[preprocess data][add_sin_cos_yaw]")
    # dataset = add_sin_cos_yaw(dataset)

    output_file_path = os.path.join(processed_folder, f"{mode}_dataset.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(dataset, f)

    if mode == "train":
        scale_stats = [get_scale_stats(v, k) for k,v in dataset.items()]
        scale_stats_df = pd.DataFrame(scale_stats)
        
        scale_stats_summary = scale_stats_df.groupby("vehicle_model").agg({
            "acc_min": min, "acc_max": max, 
            "steering_min": min, "steering_max": max
        }).reset_index()
        logging.info(f"{scale_stats_summary}")
        
        speed_stats = get_speed_stats(dataset)
        logging.info(f"{speed_stats}")
        scale_stats_summary.to_csv(os.path.join(processed_folder, "train_stats_summary.csv"), index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)

    root_folder = config.input_data.root_data_folder
    train_folder = os.path.join(root_folder, config.input_data.train_dataset_folder)
    test_folder = os.path.join(root_folder, config.input_data.test_dataset_folder)
    processed_folder = os.path.join(root_folder, config.preprocess_params.exp_data_folder)

    output_dir = os.path.join(root_folder, config.preprocess_params.exp_data_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    val_split_filename = config.preprocess_params.val_split_filename
    val_split_path = os.path.join(output_dir, val_split_filename)
    prepare_val_split(config, train_folder, val_split_path)

    if not args.only_test:
        logging.info("[preprocess data][train]")
        preprocess_data(
            config, 
            input_folder=train_folder, 
            processed_folder=output_dir, 
            val_split_path=val_split_path,
            mode="train"
        )

        logging.info("[preprocess data][val]")
        preprocess_data(
            config, 
            input_folder=train_folder, 
            processed_folder=output_dir, 
            val_split_path=val_split_path,
            mode="val"
        )

    logging.info("[preprocess data][test]")
    preprocess_data(
        config, 
        input_folder=test_folder, 
        processed_folder=processed_folder, 
        val_split_path=val_split_path,
        mode="test"
    )
