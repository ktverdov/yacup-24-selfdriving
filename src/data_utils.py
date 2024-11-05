import os
import numpy as np
import pandas as pd
from copy import deepcopy
import json

import typing
from tqdm import tqdm

from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- secs ---

NSECS_IN_SEC = 1000000000

def secs_to_nsecs(secs: float):
    return int(secs * NSECS_IN_SEC)

def nsecs_to_secs(nsecs: int):
    return float(nsecs) / NSECS_IN_SEC

def yaw_direction(yaw_value):
    return np.array([np.cos(yaw_value), np.sin(yaw_value)])

# --- read raw dataset

def read_testcase_ids(dataset_path: str):
    ids = [int(case_id) for case_id in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, case_id))]
    return ids


class DataFilePaths:
    def __init__(self, testcase_path: str):
        self.testcase_path = testcase_path
        
    def localization(self):
        return os.path.join(self.testcase_path, 'localization.csv')
    
    def control(self):
        return os.path.join(self.testcase_path, 'control.csv')
    
    def metadata(self):
        return os.path.join(self.testcase_path, 'metadata.json')
    
    # exists only for test_dataset
    def requested_stamps(self):
        return os.path.join(self.testcase_path, 'requested_stamps.csv')

def read_localization(localization_path: str):
    return pd.read_csv(localization_path)

def read_control(control_path):
    return pd.read_csv(control_path)

def read_metadata(metadata_path: str):
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    return data

def read_requested_stamps(requested_stamps_path: str):
    return pd.read_csv(requested_stamps_path)
    
def read_testcase(dataset_path: str, testcase_id: str, is_test: bool = False):
    testcase_path = os.path.join(dataset_path, str(testcase_id))
    data_file_paths = DataFilePaths(testcase_path)
    
    testcase_data = {}
    testcase_data['localization'] = read_localization(data_file_paths.localization())
    testcase_data['control'] = read_control(data_file_paths.control())
    testcase_data['metadata'] = read_metadata(data_file_paths.metadata())
    if is_test:
        testcase_data['requested_stamps'] = read_requested_stamps(data_file_paths.requested_stamps())
        
    return testcase_data

def read_testcases(dataset_path: str, is_test: bool = False, testcase_ids: typing.Iterable[int] = None):
    result = {}
    if testcase_ids is None:
        testcase_ids = read_testcase_ids(dataset_path)
    
    for testcase_id in tqdm(testcase_ids):
        testcase = read_testcase(dataset_path, testcase_id, is_test=is_test)
        result[testcase_id] = testcase
    return result

#---

localization_time_end = secs_to_nsecs(5)
control_time_end = secs_to_nsecs(20)

def get_val_gt_dataset(val_dataset):
    val_dataset_gt = dict()
    
    for testcase_id, testcase in val_dataset.items():
        testcase_df = {k: deepcopy(v) for k,v in testcase.items()}

        testcase_df["gt"] = testcase_df["localization"][(
            (localization_time_end < testcase_df["localization"]["stamp_ns"]) & 
            (testcase_df["localization"]["stamp_ns"] < control_time_end))
        ][["stamp_ns", "x", "y", "yaw"]]
        testcase_df["gt"]["testcase_id"] = testcase_id

        testcase_df["requested_stamps"] = testcase_df["gt"][["stamp_ns"]]
        testcase_df["localization"] = testcase_df["localization"][testcase_df["localization"]["stamp_ns"] < localization_time_end]
        testcase_df["control"] = testcase_df["control"][testcase_df["control"]["stamp_ns"] < control_time_end]

        val_dataset_gt[testcase_id] = testcase_df

    return val_dataset_gt

# --- val split ---

def get_validation_split(data_ids, val_size): # train_dataset
    train_ids = data_ids[:(- int(len(data_ids) * val_size) )]
    val_ids = data_ids[(- int(len(data_ids) * val_size) ):]

    return train_ids, val_ids

# ---


# --- interpolation to grid train ---

def interpolate_df_to_grid(df, time_grid, fill_value):
    time_column = "stamp_ns"
    data_interpolated = {time_column: time_grid}
    
    # Interpolate for each column (excluding time column)
    for column in df.columns:
        if column == "stamp_ns":
            continue

        interp_func = interpolate.interp1d(df[time_column], df[column], kind='linear', fill_value=fill_value, bounds_error=False)
        data_interpolated[column] = interp_func(time_grid)

    df_interpolated = pd.DataFrame(data_interpolated)
    
    return df_interpolated


def interpolate_case_to_grid(case_data, grid_size):
    time_max = case_data["localization"]["stamp_ns"].max()
    time_grid = np.arange(0, time_max, grid_size)

    df_loc_interp = interpolate_df_to_grid(case_data["localization"][["stamp_ns", "x", "y", "z", "yaw", "roll", "pitch"]], time_grid, fill_value="extrapolate")
    df_control_interp = interpolate_df_to_grid(case_data["control"][["stamp_ns", "acceleration_level", "steering"]], time_grid, fill_value=0)

    data_interp = {"localization": df_loc_interp, "control": df_control_interp, "metadata": case_data["metadata"]}

    return data_interp


def interpolate_case_to_grid_test(case_data, grid_size):
    time_max_location = secs_to_nsecs(5)
    time_grid_location = np.arange(0, time_max_location, grid_size)

    time_max_control = secs_to_nsecs(20)
    time_grid_control = np.arange(0, time_max_control, grid_size)

    df_loc_interp = interpolate_df_to_grid(case_data["localization"][["stamp_ns", "x", "y", "z", "yaw", "roll", "pitch"]], time_grid_location, fill_value="extrapolate")
    df_control_interp = interpolate_df_to_grid(case_data["control"][["stamp_ns", "acceleration_level", "steering"]], time_grid_control, fill_value=0)

    data_interp = {"localization": df_loc_interp, "control": df_control_interp, "metadata": case_data["metadata"]}

    return data_interp


def interpolate_to_test_grid(df, gt_grid, time_column="stamp_ns", last_speed=None):
    interp_func = interpolate.interp1d(
        df[time_column], df[target_column], kind='linear', fill_value="extrapolate"
    )
    speed_interp = interp_func(gt_grid)
    # diff = speed_interp[0] - last_speed
    # speed_interp = [x - diff for x in speed_interp]

    return dict(zip(gt_grid, speed_interp))

def pred2gt_interpolate(predictions, gt_dataset, target_column):
    predictions_final = dict()

    for testcase_id, v in gt_dataset.items():
        pred_df = predictions[predictions["testcase_id"] == testcase_id][["stamp_ns", target_column]]
        if "requested_stamps" in v:
            requested_stamps = v["requested_stamps"]["stamp_ns"].values
        else:
            requested_stamps = v["gt"]["stamp_ns"].values

        pred_df_interpolated = interpolate_df_to_grid(pred_df, requested_stamps, fill_value="extrapolate")
        predictions_final[testcase_id] = dict(zip(pred_df_interpolated["stamp_ns"], pred_df_interpolated[target_column]))

    return predictions_final

# ---


def add_speed_to_data(dataset, grid_size):
    for k,v in tqdm(dataset.items()):
        df_speed = v["localization"][["x", "y", "stamp_ns"]].copy()
        df_speed["distance"] = np.sqrt(df_speed["x"].diff()**2 + df_speed["y"].diff()**2)
        df_speed["speed"] = np.clip(df_speed["distance"] / nsecs_to_secs(grid_size), a_min=0, a_max=None)
        df_speed["speed"] = df_speed["speed"].shift(-1)
        df_speed.loc[df_speed.index[-1], "speed"] = df_speed["speed"].iloc[-2]
        df_speed = df_speed[["stamp_ns", "speed"]]

        v["localization"] = v["localization"].merge(df_speed, on="stamp_ns")

    return dataset


def add_sin_cos_yaw(dataset):
    for k,v in tqdm(dataset.items()):
        v["localization"]["yaw_sin"] = np.sin(v["localization"]["yaw"])
        v["localization"]["yaw_cos"] = np.cos(v["localization"]["yaw"])

    return dataset


# --- stats for scalers ---

def get_speed_stats(dataset):
    speed_min = np.inf
    speed_max = 0

    for k,v in dataset.items():
        speed_values = v["localization"]["speed"].values
        speed_min = min(speed_min, min(speed_values))
        speed_max = max(speed_max, max(speed_values))
    
    return speed_min, speed_max


def get_scale_stats(testcase_df, testcase_id):
    acc_values = testcase_df["control"]["acceleration_level"].values
    steering_values = testcase_df["control"]["steering"].values

    stats = {}
    stats["testcase_id"] = testcase_id
    stats["acc_min"] = min(acc_values)
    stats["acc_max"] = max(acc_values)
    stats["acc_median"] = np.median(acc_values)

    stats["steering_min"] = min(steering_values)
    stats["steering_max"] = max(steering_values)
    stats["steering_median"] = np.median(steering_values)
    stats["vehicle_id"] = testcase_df["metadata"]["vehicle_id"]
    stats["vehicle_model"] = testcase_df["metadata"]["vehicle_model"]

    return stats
