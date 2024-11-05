import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils import nsecs_to_secs, yaw_direction


def localization_df_to_poses(loc_df):
    poses = []
    for stamp_ns, x, y, yaw in zip(loc_df['stamp_ns'], loc_df['x'], loc_df['y'], loc_df['yaw']):
        poses.append({'stamp_ns': stamp_ns, 'pos': np.array([x, y]), 'yaw': yaw})
    return poses

# naive estimation of speed at last known localization pose
def dummy_estimate_last_speed(localization_poses):
    last_pose = localization_poses[-1]
    
    start_pose_idx = -1
    for i, pose in enumerate(localization_poses, start=1-len(localization_poses)):
        start_pose_idx = i
        if nsecs_to_secs(last_pose['stamp_ns']) - nsecs_to_secs(pose['stamp_ns']) > 1.: # sec
            break
            
    start_pose = localization_poses[start_pose_idx]
    dt_sec = nsecs_to_secs(last_pose['stamp_ns']) - nsecs_to_secs(start_pose['stamp_ns'])
    
    if dt_sec > 1e-5:
        return np.linalg.norm(last_pose['pos'][:2] - start_pose['pos'][:2]) / dt_sec
    return 5. # some default value

def dummpy_predict_pose(last_loc_pose: dict, last_speed: float, prediction_stamp: int, yaw_pred: float, speed_pred):
    dt_sec = nsecs_to_secs(prediction_stamp) - nsecs_to_secs(last_loc_pose['stamp_ns'])
    distance = dt_sec * speed_pred

    direction = yaw_direction(yaw_pred)
    pos_translate = direction * distance
    return {"pos": last_loc_pose['pos'] + pos_translate, 
            'yaw': yaw_pred,
           }

def predict_testcase(testcase: dict, yaw_preds: dict, speed_preds: dict):
    loc_df = testcase['localization']
    localization_poses = localization_df_to_poses(loc_df)
    
    last_loc_pose = localization_poses[-1]
    yaw_init = last_loc_pose["yaw"]
    last_speed = dummy_estimate_last_speed(localization_poses)
    
    predicted_poses = []
    for stamp in testcase['requested_stamps']['stamp_ns']:
        if len(predicted_poses) == 0:
            last_loc_pose = last_loc_pose
            prev_stamp = stamp
        else:
            last_loc_pose = predicted_poses[-1]
            last_loc_pose["stamp_ns"] = prev_stamp
            prev_stamp = stamp

        if yaw_preds:
            yaw_pred = yaw_preds[stamp]
        else:
            yaw_pred = yaw_init

        if speed_preds:
            speed_pred = speed_preds[stamp]
        else:
            speed_pred = last_speed

        pose = dummpy_predict_pose(last_loc_pose, last_speed, stamp, yaw_pred, speed_pred)
        predicted_poses.append(pose)
        
    predictions = {}
    predictions['stamp_ns'] = testcase['requested_stamps']['stamp_ns']
    predictions['x'] = [pose['pos'][0] for pose in predicted_poses]
    predictions['y'] = [pose['pos'][1] for pose in predicted_poses]
    predictions['yaw'] = [pose['yaw'] for pose in predicted_poses]
    return pd.DataFrame(predictions)

def predict_test_dataset(test_dataset: dict, yaw_preds: dict, speed_preds: dict):
    predictions = {}
    for testcase_id, testcase in tqdm(test_dataset.items()): 
        predictions[testcase_id] = predict_testcase(testcase, yaw_preds.get(testcase_id, {}), speed_preds.get(testcase_id, {}))
    return predictions

def get_predictions_df(dataset_predictions: dict):
    prediction_list = []
    for testcase_id, prediction in tqdm(dataset_predictions.items()):
        prediction['testcase_id'] = [testcase_id] * len(prediction)
        prediction_list.append(prediction)
    predictions_df = pd.concat(prediction_list)
    predictions_df = predictions_df.reindex(columns=["testcase_id", "stamp_ns", "x", "y", "yaw"])

    return predictions_df



#----

def dummpy_predict_pose_dxdy(last_loc_pose: dict, last_speed: float, prediction_stamp: int, yaw_pred: float, dxdy_pred):
    dt_sec = nsecs_to_secs(prediction_stamp) - nsecs_to_secs(last_loc_pose['stamp_ns'])
    dxdy_pred = np.array([dt_sec * dxdy_pred[0], dt_sec * dxdy_pred[1]])

    direction = yaw_direction(yaw_pred)
    pos_translate = dxdy_pred
    return {"pos": last_loc_pose['pos'] + pos_translate, 
            'yaw': yaw_pred,
           }


def predict_testcase_dxdy(testcase: dict, yaw_preds: dict, dxdy_preds: dict):
    loc_df = testcase['localization']
    localization_poses = localization_df_to_poses(loc_df)
    
    last_loc_pose = localization_poses[-1]
    yaw_init = last_loc_pose["yaw"]
    last_speed = dummy_estimate_last_speed(localization_poses)
    
    predicted_poses = []
    for stamp in testcase['requested_stamps']['stamp_ns']:
        if len(predicted_poses) == 0:
            last_loc_pose = last_loc_pose
            prev_stamp = stamp
        else:
            last_loc_pose = predicted_poses[-1]
            last_loc_pose["stamp_ns"] = prev_stamp
            prev_stamp = stamp

        if yaw_preds:
            yaw_pred = yaw_preds[stamp]
        else:
            yaw_pred = yaw_init

        if dxdy_preds:
            dxdy_pred = dxdy_preds[stamp]

        pose = dummpy_predict_pose_dxdy(last_loc_pose, last_speed, stamp, yaw_pred, dxdy_pred)
        predicted_poses.append(pose)
        
    predictions = {}
    predictions['stamp_ns'] = testcase['requested_stamps']['stamp_ns']
    predictions['x'] = [pose['pos'][0] for pose in predicted_poses]
    predictions['y'] = [pose['pos'][1] for pose in predicted_poses]
    predictions['yaw'] = [pose['yaw'] for pose in predicted_poses]
    return pd.DataFrame(predictions)

def predict_test_dataset_dxdy(test_dataset: dict, yaw_preds: dict, dxdy_preds: dict):
    predictions = {}
    for testcase_id, testcase in tqdm(test_dataset.items()): 
        predictions[testcase_id] = predict_testcase_dxdy(testcase, yaw_preds.get(testcase_id, {}), dxdy_preds.get(testcase_id, {}))
    return predictions