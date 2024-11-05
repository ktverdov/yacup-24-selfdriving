import random
import pickle
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from data_utils import secs_to_nsecs, nsecs_to_secs


def get_dataset(dataset_type, **params):
    if dataset_type == "TrajectoryDataset":
        return TrajectoryDataset(**params)
    elif dataset_type == "TrajectoryDatasetSpeedDiff":
        return TrajectoryDatasetSpeedDiff(**params)
    elif dataset_type == "TrajectoryDatasetFullData":
        return TrajectoryDatasetFullData(**params)
    elif dataset_type == "TrajectoryDatasetFullDataV0":
        return TrajectoryDatasetFullDataV0(**params)
    elif dataset_type == "TrajectoryDatasetFullDxDyData":
        return TrajectoryDatasetFullDxDyData(**params)
    elif dataset_type == "TrajectoryDatasetFullDataSpeedUnscaled":
        return TrajectoryDatasetFullDataSpeedUnscaled(**params)


class TrajectoryDataset(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        # if self.mode == "val":
        #     self.n = len(self.data)
        # else:
        #     self.n = n
        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        # if self.mode == "val":
        #     case_id = self.available_case_ids[idx]
        # else:
            # case_id = random.choice(self.available_case_ids)
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()

        if "yaw_sin" not in loc_input.columns:
            loc_input["yaw_sin"] = np.sin(loc_input["yaw"])
            loc_input["yaw_cos"] = np.cos(loc_input["yaw"])

        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        loc_input_train = loc_input_train[self.target_columns]
        loc_input_val = loc_input_val[self.target_columns]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification']]
        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}


# class TrajectoryDatasetSpeedDiff(Dataset):
#     def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
#         print(data_path)
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)
#         self.mode = mode
#         self.scalers_dict = scalers_dict

#         self.target_columns = target_columns
#         self.grid_size = grid_size

#         self.n = len(self.data)

#         self.available_case_ids = list(self.data.keys())

#     def __len__(self):
#         return self.n

#     def __getitem__(self, idx):
#         case_id = self.available_case_ids[idx]
#         data_case = self.data[case_id]

#         vehicle_model = data_case["metadata"]["vehicle_model"]

#         time_max = data_case["localization"]["stamp_ns"].max()
#         time_grid = np.arange(0, time_max, self.grid_size)

#         if self.mode == "train":
#             sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
#         else:
#             sample_start = 0

#         sample_end = sample_start + secs_to_nsecs(20)

#         loc_df = data_case["localization"].copy()
#         control_df = data_case["control"].copy()

#         loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
#                             (loc_df["stamp_ns"] < sample_end)].copy()
        
#         loc_input["speed"] = self.scalers_dict.transform(
#             "speed", 
#             loc_input["speed"].values
#         )
#         loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
#         loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
#         loc_input_train = loc_input_train[self.target_columns]
#         loc_input_val = loc_input_val[self.target_columns]

#         control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
#                                          (control_df["stamp_ns"] < sample_end)].copy()
#         control_input["acceleration_level"] = self.scalers_dict.transform(
#             f"{vehicle_model}_acc", 
#             control_input["acceleration_level"].values
#         )
#         control_input["steering"] = self.scalers_dict.transform(
#             f"{vehicle_model}_steering", 
#             control_input["steering"].values
#         )
#         control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
#         control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
#         control_input_train = control_input_train[["acceleration_level", "steering"]]
#         control_input_val = control_input_val[["acceleration_level", "steering"]]

#         history = np.hstack([loc_input_train, control_input_train])
#         future = control_input_val.values
#         y = loc_input_val.values

#         static_data = [data_case["metadata"][k] 
#                            for k in ['vehicle_model', 'vehicle_model_modification']]
#         static_data = np.array(static_data)

#         return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}


class TrajectoryDatasetFullData(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_df["speed_z"] = loc_df["z"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["speed_z"] = loc_df["speed_z"].shift(-1)
        loc_df.loc[loc_df.index[-1], "speed_z"] = loc_df["speed_z"].iloc[-2]

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()
        
        for angle_type in ["yaw", "roll", "pitch"]:
            loc_input[f"{angle_type}_sin"] = np.sin(loc_input[angle_type])
            loc_input[f"{angle_type}_cos"] = np.cos(loc_input[angle_type])

        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input["speed_x"] = loc_input["speed"] * loc_input["yaw_cos"]
        loc_input["speed_y"] = loc_input["speed"] * loc_input["yaw_sin"]

        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]

        loc_train_columns = ["speed", "yaw_sin", "yaw_cos", "roll", "pitch", "roll_sin", "roll_cos", 
                                "pitch_sin", "pitch_cos", "speed_x", "speed_y", "speed_z"]
        loc_input_train = loc_input_train[loc_train_columns]
        loc_input_val = loc_input_val[self.target_columns]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification', 'location_reference_point_id']]
        month = int(data_case["metadata"]["ride_date"].split("-")[1])
        month -= 1
        tires = data_case["metadata"]["tires"]
        tires = [tires["front"], tires["rear"]]
        static_data = static_data + tires + [month]

        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}


class TrajectoryDatasetFullDataV0(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_df["speed_z"] = loc_df["z"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["speed_z"] = loc_df["speed_z"].shift(-1)
        loc_df.loc[loc_df.index[-1], "speed_z"] = loc_df["speed_z"].iloc[-2]

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()
        
        for angle_type in ["yaw", "roll", "pitch"]:
            loc_input[f"{angle_type}_sin"] = np.sin(loc_input[angle_type])
            loc_input[f"{angle_type}_cos"] = np.cos(loc_input[angle_type])

        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input["speed_x"] = loc_input["speed"] * loc_input["yaw_cos"]
        loc_input["speed_y"] = loc_input["speed"] * loc_input["yaw_sin"]

        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]

        loc_train_columns = ["speed", "yaw_sin", "yaw_cos", "roll", "pitch", "roll_sin", "roll_cos", 
                                "pitch_sin", "pitch_cos", "speed_x", "speed_y", "speed_z"]
        loc_input_train = loc_input_train[loc_train_columns]
        loc_input_val = loc_input_val[self.target_columns]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification', 'location_reference_point_id']]
        month = int(data_case["metadata"]["ride_date"].split("-")[1])
        tires = data_case["metadata"]["tires"]
        tires = [tires["front"], tires["rear"]]
        static_data = static_data + tires + [month]

        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}


class TrajectoryDatasetFullDataSpeedUnscaled(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_df["speed_z"] = loc_df["z"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["speed_z"] = loc_df["speed_z"].shift(-1)
        loc_df.loc[loc_df.index[-1], "speed_z"] = loc_df["speed_z"].iloc[-2]

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()
        
        for angle_type in ["yaw", "roll", "pitch"]:
            loc_input[f"{angle_type}_sin"] = np.sin(loc_input[angle_type])
            loc_input[f"{angle_type}_cos"] = np.cos(loc_input[angle_type])

        loc_input["speed_original"] = loc_input["speed"]
        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input["speed_x"] = loc_input["speed"] * loc_input["yaw_cos"]
        loc_input["speed_y"] = loc_input["speed"] * loc_input["yaw_sin"]

        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]

        loc_train_columns = ["speed", "yaw_sin", "yaw_cos", "roll", "pitch", "roll_sin", "roll_cos", 
                                "pitch_sin", "pitch_cos", "speed_x", "speed_y", "speed_z"]
        loc_input_train = loc_input_train[loc_train_columns]
        loc_input_val = loc_input_val[["speed_original"]]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification', 'location_reference_point_id']]
        month = int(data_case["metadata"]["ride_date"].split("-")[1])
        month -= 1
        tires = data_case["metadata"]["tires"]
        tires = [tires["front"], tires["rear"]]
        static_data = static_data + tires + [month]

        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}


class TrajectoryDatasetFullDxDyData(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_df["speed_z"] = loc_df["z"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["speed_z"] = loc_df["speed_z"].shift(-1)
        loc_df.loc[loc_df.index[-1], "speed_z"] = loc_df["speed_z"].iloc[-2]

        loc_df["dx"] = loc_df["x"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["dx"] = loc_df["dx"].shift(-1)
        loc_df.loc[loc_df.index[-1], "dx"] = loc_df["dx"].iloc[-2]

        loc_df["dy"] = loc_df["y"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["dy"] = loc_df["dy"].shift(-1)
        loc_df.loc[loc_df.index[-1], "dy"] = loc_df["dy"].iloc[-2]

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()
        
        for angle_type in ["yaw", "roll", "pitch"]:
            loc_input[f"{angle_type}_sin"] = np.sin(loc_input[angle_type])
            loc_input[f"{angle_type}_cos"] = np.cos(loc_input[angle_type])

        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input["speed_x"] = loc_input["speed"] * loc_input["yaw_cos"]
        loc_input["speed_y"] = loc_input["speed"] * loc_input["yaw_sin"]

        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]

        loc_train_columns = ["dx", "dy", "speed", "yaw_sin", "yaw_cos", "roll", "pitch", "roll_sin", "roll_cos", 
                                "pitch_sin", "pitch_cos", "speed_x", "speed_y", "speed_z"]
        loc_input_train = loc_input_train[loc_train_columns]
        loc_input_val = loc_input_val[self.target_columns]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification', 'location_reference_point_id']]
        month = int(data_case["metadata"]["ride_date"].split("-")[1])
        tires = data_case["metadata"]["tires"]
        tires = [tires["front"], tires["rear"]]
        static_data = static_data + tires + [month]

        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}

class TrajectoryDatasetFullDataSpeedDiff(Dataset):
    def __init__(self, data_path, scalers_dict, grid_size, target_columns, mode="train", n=None):
        print(data_path)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.mode = mode
        self.scalers_dict = scalers_dict

        self.target_columns = target_columns
        self.grid_size = grid_size

        self.n = len(self.data)

        self.available_case_ids = list(self.data.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        case_id = self.available_case_ids[idx]
        data_case = self.data[case_id]

        vehicle_model = data_case["metadata"]["vehicle_model"]

        time_max = data_case["localization"]["stamp_ns"].max()
        time_grid = np.arange(0, time_max, self.grid_size)

        if self.mode == "train":
            sample_start = random.choice(np.arange(0, max(data_case["localization"]["stamp_ns"]) - secs_to_nsecs(20), self.grid_size))
        else:
            sample_start = 0

        sample_end = sample_start + secs_to_nsecs(20)

        loc_df = data_case["localization"].copy()
        control_df = data_case["control"].copy()

        loc_df["speed_z"] = loc_df["z"].diff() / nsecs_to_secs(self.grid_size)
        loc_df["speed_z"] = loc_df["speed_z"].shift(-1)
        loc_df.loc[loc_df.index[-1], "speed_z"] = loc_df["speed_z"].iloc[-2]

        loc_input = loc_df[(loc_df["stamp_ns"] >= sample_start) & 
                            (loc_df["stamp_ns"] < sample_end)].copy()
        
        for angle_type in ["yaw", "roll", "pitch"]:
            loc_input[f"{angle_type}_sin"] = np.sin(loc_input[angle_type])
            loc_input[f"{angle_type}_cos"] = np.cos(loc_input[angle_type])

        loc_input["speed"] = self.scalers_dict.transform(
            "speed", 
            loc_input["speed"].values
        )
        loc_input["speed_x"] = loc_input["speed"] * loc_input["yaw_cos"]
        loc_input["speed_y"] = loc_input["speed"] * loc_input["yaw_sin"]

        loc_input_train = loc_input[loc_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        loc_input_val = loc_input[loc_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]

        loc_train_columns = ["yaw_sin", "yaw_cos", "roll", "pitch", "roll_sin", "roll_cos", 
                                "pitch_sin", "pitch_cos", "speed_z"]
        loc_input_train = loc_input_train[loc_train_columns]
        loc_input_val = loc_input_val[self.target_columns]

        control_input = control_df[(control_df["stamp_ns"] >= sample_start) &
                                         (control_df["stamp_ns"] < sample_end)].copy()
        control_input["acceleration_level"] = self.scalers_dict.transform(
            f"{vehicle_model}_acc", 
            control_input["acceleration_level"].values
        )
        control_input["steering"] = self.scalers_dict.transform(
            f"{vehicle_model}_steering", 
            control_input["steering"].values
        )
        control_input_train = control_input[control_input["stamp_ns"] < sample_start + secs_to_nsecs(5)]
        control_input_val = control_input[control_input["stamp_ns"] >= sample_start + secs_to_nsecs(5)]
        control_input_train = control_input_train[["acceleration_level", "steering"]]
        control_input_val = control_input_val[["acceleration_level", "steering"]]

        history = np.hstack([loc_input_train, control_input_train])
        future = control_input_val.values
        y = loc_input_val.values

        static_data = [data_case["metadata"][k] 
                           for k in ['vehicle_model', 'vehicle_model_modification', 'location_reference_point_id']]
        month = int(data_case["metadata"]["ride_date"].split("-")[1])
        month -= 1
        tires = data_case["metadata"]["tires"]
        tires = [tires["front"], tires["rear"]]
        static_data = static_data + tires + [month]

        static_data = np.array(static_data)

        return {"history": history, "future": future, "y": y, "case_id": case_id, "static": static_data}