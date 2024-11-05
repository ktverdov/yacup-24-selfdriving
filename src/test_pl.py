import os
import yaml
from addict import Dict
import pandas as pd
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from normalize_data import SCALERS_DICT
from datasets import get_dataset
from pl_modules import TrajectoryPredictorSpeed, TrajectoryPredictorYaw, TrajectoryPredictorTFT, TrajectoryPredictorTFTQuantile
from vehicle_model import predict_test_dataset, get_predictions_df
from data_utils import secs_to_nsecs, pred2gt_interpolate, get_val_gt_dataset, read_testcases
from vehicle_model import predict_test_dataset
from test_metrics import calculate_metric_dataset


class TestPipeline:
    def __init__(self, speed_model, yaw_model, data_config, train_config_speed, train_config_yaw, mode="val"):
        self.mode = mode

        self.speed_model = speed_model
        self.yaw_model = yaw_model

        self.data_config = data_config
        self.grid_size = data_config.preprocess_params.grid_size

        if self.speed_model is not None:
            self.dataloader_speed = self._get_dataloader(data_config, train_config_speed, mode)
    
        if self.yaw_model is not None:
            self.dataloader_yaw = self._get_dataloader(data_config, train_config_yaw, mode)

    def _get_dataloader(self, data_config, train_config, mode):
        dataset = get_dataset(
            dataset_type=train_config.dataset.type,
            data_path=os.path.join(
                data_config.input_data.root_data_folder, 
                data_config.preprocess_params.exp_data_folder, 
                f"{mode}_dataset.pkl"
            ),
            scalers_dict=SCALERS_DICT,
            target_columns=train_config.dataset.target_columns,
            grid_size=data_config.preprocess_params.grid_size,
            mode="val",
        )
        print(len(dataset))

        dataloader = DataLoader(dataset, batch_size=train_config.dataset.train_bs, num_workers=8, shuffle=False)

        return dataloader

    def _get_model_predictions(self, model, dataloader, target_column):
        model.eval()
        predictions = []
        case_ids = []
        stamps = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output = model({k:v.cuda() for k,v in batch.items()})

                predictions.extend(output.cpu().numpy().reshape(-1, output.shape[-1]))
                case_ids.extend([cid for cid in batch["case_id"].cpu().numpy() for _ in range(output.shape[1])])
                stamps.extend([t for _ in range(len(batch["case_id"])) for t in range(output.shape[1])])

        if target_column == "speed":
            predictions = np.array(predictions).squeeze()
            predictions = np.clip(predictions, a_min=0, a_max=None)
        elif target_column == "yaw":
            predictions = np.array(predictions)
            predictions = np.arctan2(predictions[:, 0], predictions[:, 1])

        if target_column in SCALERS_DICT.scalers_dict.keys():
            predictions = SCALERS_DICT.inverse_transform(target_column, predictions)

        df = pd.DataFrame({
            "testcase_id": case_ids,
            "stamp_ns": stamps,
            f"{target_column}": predictions
        })
        return df

    def _postprocess_predictions(self, preds):
        pass

    def predict_trajectories(self, output_path):
        if self.mode == "val":
            ids = joblib.load(os.path.join(
                self.data_config.input_data.root_data_folder,
                self.data_config.preprocess_params.exp_data_folder, 
                self.data_config.preprocess_params.val_split_filename)
            )["val_ids"]
            input_folder = os.path.join(self.data_config.input_data.root_data_folder, self.data_config.input_data.train_dataset_folder)

            initial_dataset = read_testcases(input_folder, testcase_ids=ids)
            initial_dataset = get_val_gt_dataset(initial_dataset)
        elif self.mode == "test":
            test_dataset_path = os.path.join(self.data_config.input_data.root_data_folder, self.data_config.input_data.test_dataset_folder)
            initial_dataset = read_testcases(test_dataset_path, is_test=True)

        if self.speed_model is not None:
            target_column = "speed"
            speed_preds = self._get_model_predictions(self.speed_model, self.dataloader_speed, target_column)
            speed_preds["stamp_ns"] = self.grid_size * speed_preds["stamp_ns"] + secs_to_nsecs(5)
            speed_preds = pred2gt_interpolate(speed_preds, initial_dataset, target_column)
        else:
            speed_preds = {}

        if self.yaw_model is not None:
            target_column = "yaw"
            yaw_preds = self._get_model_predictions(self.yaw_model, self.dataloader_yaw, target_column)
            yaw_preds["stamp_ns"] = self.grid_size * yaw_preds["stamp_ns"] + secs_to_nsecs(5)
            yaw_preds = pred2gt_interpolate(yaw_preds, initial_dataset, target_column)
        else:
            # print("GT YAW used")
            # yaw_preds = {k:dict(zip(v["gt"]["stamp_ns"], v["gt"]["yaw"])) for k,v in initial_dataset.items()}
            yaw_preds = {}

        if self.speed_model is not None:
            joblib.dump(speed_preds, f"predictions/{output_path}_{self.mode}_speed.dump")

        if self.yaw_model is not None:
            joblib.dump(yaw_preds, f"predictions/{output_path}_{self.mode}_yaw.dump")

        return None, None
        # predictions = predict_test_dataset(initial_dataset, yaw_preds, speed_preds)
        # predictions_df = get_predictions_df(predictions)

        # return initial_dataset, predictions_df


    def evaluate_validation(self, output_path):
        initial_dataset, predictions_df = self.predict_trajectories(output_path)
        
        # val_trues = {k: v["gt"] for k, v in initial_dataset.items()}
        # val_trues_df = get_predictions_df(val_trues)
        
        # metric_df, mean_metric = calculate_metric_dataset(val_trues_df, predictions_df)

        # val_info = {
        #     'predictions': predictions_df,
        #     'ground_truth': val_trues_df,
        #     'metrics': metric_df,
        #     'mean_metric': mean_metric
        # }
        # joblib.dump(val_info, f"predictions/{output_path}_{self.mode}_predictions.dump")
        # return val_info
        return {"mean_metric": 0}

    def predict_test(self, output_filepath):
        _, test_predictions = self.predict_trajectories(output_filepath)
        # test_predictions.to_csv(f"predictions/{output_filepath}.csv.gz", index=False, header=True, compression='gzip')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=False)

    parser.add_argument('--train_config_speed', type=str, required=False)
    parser.add_argument('--speed_checkpoint', type=str, required=False)

    parser.add_argument('--train_config_yaw', type=str, required=False)
    parser.add_argument('--yaw_checkpoint', type=str, required=False)

    parser.add_argument('--run_test', type=bool, required=False, default=False)

    parser.add_argument('--output_path', type=str, required=False, default="temp")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Dict(config)


if __name__ == "__main__":
    args = parse_args()
    data_config = load_config(args.data_config)
    if args.train_config_speed is not None:
        train_config_speed = load_config(args.train_config_speed)
    else:
        train_config_speed = None
    speed_checkpoint = args.speed_checkpoint
    print(speed_checkpoint)

    if args.train_config_yaw is not None:
        train_config_yaw = load_config(args.train_config_yaw)
    else:
        train_config_yaw = None
    yaw_checkpoint = args.yaw_checkpoint

    run_test = args.run_test

    if speed_checkpoint:
        speed_model = TrajectoryPredictorTFT.load_from_checkpoint(speed_checkpoint, config=train_config_speed, scalers_dict=SCALERS_DICT)
    else:
        speed_model = None

    if yaw_checkpoint:
        yaw_model = TrajectoryPredictorYaw.load_from_checkpoint(yaw_checkpoint, config=train_config_yaw, scalers_dict=SCALERS_DICT)
    else:
        yaw_model = None

    if not run_test:
        test_pl = TestPipeline(
            speed_model,
            yaw_model,
            data_config, 
            train_config_speed,
            train_config_yaw,
            mode="val"
        )

        val_results = test_pl.evaluate_validation(args.output_path)
        print(f"Validation mean metric: {val_results['mean_metric']}")

    if run_test:
        test_pl = TestPipeline(
            speed_model,
            yaw_model,
            data_config, 
            train_config_speed,
            train_config_yaw,
            mode="test"
        )

        test_pl.predict_test(args.output_path)
