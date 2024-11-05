import os
import yaml
import joblib
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import argparse

from data_utils import read_testcases, get_val_gt_dataset
from vehicle_model import predict_test_dataset, get_predictions_df
from test_metrics import calculate_metric_dataset


def calculate_mean_across_submissions(submissions):
    cumulative_values = defaultdict(lambda: defaultdict(list))

    for submission in tqdm(submissions):
        for testcase, time_values in submission.items():
            for time, value in time_values.items():
                cumulative_values[testcase][time].append(value)

    mean_values = {}
    for testcase, time_values in cumulative_values.items():
        mean_values[testcase] = {time: np.mean(values) for time, values in time_values.items()}
    
    return cumulative_values, mean_values


def main(args):
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    sub_path = "./predictions/"
    
    speed_sub_names = [f"speed_{i}_test_speed.dump" for i in range(5)]
    yaw_sub_names = [f"yaw_{i}_test_yaw.dump" for i in range(5)]

    speed_preds_list = [
        joblib.load(os.path.join(sub_path, sub_name))
        for sub_name in speed_sub_names
    ]

    _, speed_means = calculate_mean_across_submissions(speed_preds_list)

    yaw_preds_list = [
        joblib.load(os.path.join(sub_path, sub_name))
        for sub_name in yaw_sub_names
    ]

    _, yaw_means = calculate_mean_across_submissions(yaw_preds_list)

    test_dataset_path = os.path.join(data_config['input_data']['root_data_folder'], 
                                    data_config['input_data']['test_dataset_folder'])
    initial_dataset = read_testcases(test_dataset_path, is_test=True)

    predictions = predict_test_dataset(initial_dataset, yaw_means, speed_means)
    predictions_df = get_predictions_df(predictions)

    output_filepath = "sub_final"
    predictions_df.to_csv(
        f"./predictions/{output_filepath}.csv.gz", 
        index=False, 
        header=True, 
        compression='gzip'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    args = parser.parse_args()
    main(args)
