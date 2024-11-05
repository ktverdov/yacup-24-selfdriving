import numpy as np
import pandas as pd

SEGMENT_LENGTH = 1.

def yaw_direction(yaw_value):
    return np.array([np.cos(yaw_value), np.sin(yaw_value)])

def build_car_points(x_y_yaw):
    directions = np.vstack(yaw_direction(x_y_yaw[:, -1]))
    
    front_points = x_y_yaw[:, :-1] + SEGMENT_LENGTH * directions.T
    points = np.vstack([x_y_yaw[:, :-1], front_points])
    return points

def build_car_points_from_merged_df(df: pd.DataFrame):
    points_gt = df[['x_gt', 'y_gt', 'yaw_gt']].to_numpy()
    points_pred = df[['x_pred', 'y_pred', 'yaw_pred']].to_numpy()
    
    points_gt = build_car_points(points_gt)
    points_pred = build_car_points(points_pred)
    return points_gt, points_pred

def calculate_metric_testcase(df: pd.DataFrame):        
    points_gt, points_pred = build_car_points_from_merged_df(df)
    
    metric = np.mean(np.sqrt(2 * np.mean((points_gt - points_pred) ** 2, axis=1)))
    return metric

def calculate_metric_dataset(ground_truth_df: pd.DataFrame, prediction_df: pd.DataFrame):
    assert (len(ground_truth_df) == len(prediction_df))
    
    df = ground_truth_df.merge(prediction_df, on=['testcase_id', 'stamp_ns'], suffixes=['_gt', '_pred'])
    
    metric = df.groupby('testcase_id').apply(calculate_metric_testcase)
    return metric, np.mean(metric)
