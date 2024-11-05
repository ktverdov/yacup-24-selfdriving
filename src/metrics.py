import numpy as np
import pandas as pd


class Metric():
    def __init__(self):
        self.input_type = None
    
    def update(self):
        pass
    
    def compute(self):
        pass


def get_val_metric(val_metric_type, scalers_dict):
    if val_metric_type == "speed_mse":
        return TrajectoryMetricSpeed(scalers_dict)
    elif val_metric_type == "yaw_mse":
        return TrajectoryMetricYaw()
    elif val_metric_type == "speed_mse_unscaled":
        return TrajectoryMetricSpeedUnscaled(scalers_dict)
    return None


class TrajectoryMetricSpeed(Metric):
    def __init__(self, scalers_dict):
        self.scalers_dict = scalers_dict
        self.reinit()

    def update(self, batch, output):
        for i, case_id in enumerate(batch["case_id"].numpy()):
            y_batch_true = batch["y"].numpy()[i]
            y_true_case = self.scalers_dict.inverse_transform("speed", y_batch_true)
            y_batch_pred = output.detach().numpy()[i]
            y_pred_case = self.scalers_dict.inverse_transform("speed", y_batch_pred)
            self.val_y_true.extend(y_true_case)
            self.val_y_pred.extend(y_pred_case)
            self.val_case_ids.extend([case_id] * len(y_pred_case))
            self.val_stamp_ns.extend([_ for _ in range(len(y_pred_case))])
            
    def compute(self):
        val_y_true_df = pd.DataFrame(self.val_y_true, columns=["speed"])
        val_y_true_df["testcase_id"] = self.val_case_ids
        val_y_true_df["stamp_ns"] = self.val_stamp_ns
        
        val_y_pred_df = pd.DataFrame(self.val_y_pred, columns=["speed"])
        val_y_pred_df["testcase_id"] = self.val_case_ids
        val_y_pred_df["stamp_ns"] = self.val_stamp_ns

        self.val_y_pred_df = val_y_pred_df
        self.val_y_true_df = val_y_true_df

        res = self.val_y_true_df.merge(self.val_y_pred_df, on=["testcase_id", "stamp_ns"], suffixes=["_gt", "_pred"])
        res["mse"] = np.sqrt((res["speed_gt"] - res["speed_pred"]) ** 2)
        mse_metric = res["mse"].mean()

        import joblib
        joblib.dump(res, "speed_preds_temp.dump")

        return {"mse": mse_metric}

    def reinit(self):
        self.val_y_true = []
        self.val_y_pred = []
        self.val_case_ids = []
        self.val_stamp_ns = []


class TrajectoryMetricSpeedUnscaled(Metric):
    def __init__(self, scalers_dict):
        self.scalers_dict = scalers_dict
        self.reinit()

    def update(self, batch, output):
        for i, case_id in enumerate(batch["case_id"].numpy()):
            y_batch_true = batch["y"].numpy()[i]
            y_true_case = y_batch_true
            y_batch_pred = output.detach().numpy()[i]
            y_pred_case = y_batch_pred
            self.val_y_true.extend(y_true_case)
            self.val_y_pred.extend(y_pred_case)
            self.val_case_ids.extend([case_id] * len(y_pred_case))
            self.val_stamp_ns.extend([_ for _ in range(len(y_pred_case))])
            
    def compute(self):
        val_y_true_df = pd.DataFrame(self.val_y_true, columns=["speed"])
        val_y_true_df["testcase_id"] = self.val_case_ids
        val_y_true_df["stamp_ns"] = self.val_stamp_ns
        
        val_y_pred_df = pd.DataFrame(self.val_y_pred, columns=["speed"])
        val_y_pred_df["testcase_id"] = self.val_case_ids
        val_y_pred_df["stamp_ns"] = self.val_stamp_ns

        self.val_y_pred_df = val_y_pred_df
        self.val_y_true_df = val_y_true_df

        res = self.val_y_true_df.merge(self.val_y_pred_df, on=["testcase_id", "stamp_ns"], suffixes=["_gt", "_pred"])
        res["mse"] = np.sqrt((res["speed_gt"] - res["speed_pred"]) ** 2)
        mse_metric = res["mse"].mean()

        import joblib
        joblib.dump(res, "speed_preds_temp.dump")

        return {"mse": mse_metric}

    def reinit(self):
        self.val_y_true = []
        self.val_y_pred = []
        self.val_case_ids = []
        self.val_stamp_ns = []


class TrajectoryMetricYaw(Metric):
    def __init__(self):
        self.reinit()

    def update(self, batch, output):
        for i, case_id in enumerate(batch["case_id"].numpy()):
            y_batch_true = batch["y"].numpy()[i]
            y_true_case = np.arctan2(y_batch_true[:, 0], y_batch_true[:, 1])
            self.val_y_true_sincos.extend(y_batch_true)

            y_batch_pred = output.detach().numpy()[i]
            y_pred_case = np.arctan2(y_batch_pred[:, 0], y_batch_pred[:, 1])
            self.val_y_pred_sincos.extend(y_batch_pred)

            self.val_y_true.extend(y_true_case)
            self.val_y_pred.extend(y_pred_case)
            self.val_case_ids.extend([case_id] * len(y_true_case))
            self.val_stamp_ns.extend([_ for _ in range(len(y_true_case))])
            
    def compute(self):
        val_y_true_df = pd.DataFrame(self.val_y_true, columns=["yaw"])
        val_y_true_df["testcase_id"] = self.val_case_ids
        val_y_true_df["stamp_ns"] = self.val_stamp_ns
        
        val_y_pred_df = pd.DataFrame(self.val_y_pred, columns=["yaw"])
        val_y_pred_df["testcase_id"] = self.val_case_ids
        val_y_pred_df["stamp_ns"] = self.val_stamp_ns

        self.val_y_pred_df = val_y_pred_df
        self.val_y_true_df = val_y_true_df

        res_dict = {}
        res_dict["merged"] = self.val_y_true_df.merge(self.val_y_pred_df, on=["testcase_id", "stamp_ns"], suffixes=["_gt", "_pred"])
        res_dict["true"] = val_y_true_df
        res_dict["pred"] = val_y_pred_df
        res = res_dict["merged"]

        import joblib
        joblib.dump(res_dict, "yaw_preds_temp.dump")

        res["mse"] = np.sqrt((res["yaw_gt"] - res["yaw_pred"]) ** 2)
        mse_metric = res["mse"].mean()


        cos_sim = [1 - np.dot(p, t) for p, t in zip(self.val_y_pred_sincos, self.val_y_true_sincos)]
        cos_sim_metric = np.mean(cos_sim)


        return {"mse": mse_metric, "cos_sim": cos_sim_metric}

    def reinit(self):
        self.val_y_true = []
        self.val_y_true_sincos = []
        self.val_y_pred = []
        self.val_y_pred_sincos = []
        self.val_case_ids = []
        self.val_stamp_ns = []
