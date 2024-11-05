from addict import Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from criterions import get_loss
from models import get_model
from metrics import get_val_metric


def get_pl_module(config, scalers_dict):
    if config.pl_model == "TrajectoryPredictorSpeed":
        return TrajectoryPredictorSpeed(config, scalers_dict)
    elif config.pl_model == "TrajectoryPredictorYaw":
        return TrajectoryPredictorYaw(config, scalers_dict)
    elif config.pl_model == "TrajectoryPredictorSpeedReluUnscaled":
        return TrajectoryPredictorSpeedReluUnscaled(config, scalers_dict)
    elif config.pl_model == "TrajectoryPredictorTFT":
        return TrajectoryPredictorTFT(config, scalers_dict)
    elif config.pl_model == "TrajectoryPredictorTFTQuantile":
        return TrajectoryPredictorTFTQuantile(config, scalers_dict)
    elif config.pl_model == "TrajectoryPredictorTFTYaw":
        return TrajectoryPredictorTFTYaw(config, scalers_dict)
    else:
        # return TrajectoryPredictorDxDy(config, scalers_dict)
        return None

class TrajectoryPredictor(pl.LightningModule):
    def __init__(self, config, scalers_dict):
        super().__init__()
        self.config = config
        self.model = get_model(config.model)        
        self.criterion = get_loss(config.loss)

        self.val_metric = get_val_metric(config.dataset.get("val_metric", None), scalers_dict)
        print(self.val_metric)

        # self.save_hyperparameters()

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = self.criterion(output, batch["y"].float())
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        loss = self.criterion(output, batch["y"].float())
        
        if self.val_metric:
            self.val_metric.update({k:v.cpu() for k,v in batch.items()}, output.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_metric:
            metric_values_dict = self.val_metric.compute()
            for name, metric_value in metric_values_dict.items():
                self.log(f"val_metric_{name}", metric_value)

            self.val_metric.reinit()

    def configure_optimizers(self):
        if self.config.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                **self.config.optimizer.parameters
            )
        return optimizer

class TrajectoryPredictorDxDy(TrajectoryPredictor):
    def forward(self, batch):
        output = self.model(
            x_hist=batch["history"][:, :, :2].float(),
            x_extra_hist=batch["history"][:, :, 2:].float(),
            x_extra_future=batch["future"].float(),
            x_static=batch["static"].float()
        )

        return output


class TrajectoryPredictorSpeed(TrajectoryPredictor):
    def forward(self, batch):
        model_type = self.config.model.type
        if model_type == "tsmixer_ext":
            output = self.model(
                x_hist=batch["history"][:, :, 0].unsqueeze(-1).float(),
                x_extra_hist=batch["history"][:, :, 1:].float(),
                x_extra_future=batch["future"].float(),
                x_static=batch["static"].float()
            )
        elif model_type == "transformer":
            output = self.model(
                history_input=batch["history"].float(),
                future_input=batch["future"].float(),
                static_input=batch["static"].float()
            )

        return output

class TrajectoryPredictorSpeedReluUnscaled(TrajectoryPredictor):
    def forward(self, batch):
        output = self.model(
            x_hist=batch["history"][:, :, 0].unsqueeze(-1).float(),
            x_extra_hist=batch["history"][:, :, 1:].float(),
            x_extra_future=batch["future"].float(),
            x_static=batch["static"].float()
        )
        output = nn.ReLU()(output)
        max_speed = 55.0
        output = output * max_speed

        return output


class TrajectoryPredictorYaw(TrajectoryPredictor):
    def forward(self, batch):
        output = self.model(
            x_hist=batch["history"][:, :, :2].float(),
            x_extra_hist=batch["history"][:, :, 2:].float(),
            x_extra_future=batch["future"].float(),
            x_static=batch["static"].float()
        )

        output = F.normalize(output, p=2.0, dim=2)

        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        output_flat = output.reshape(-1, 2)
        y = batch["y"].reshape(-1, 2).float()

        loss = self.criterion(output_flat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        output_flat = output.reshape(-1, 2)
        y = batch["y"].reshape(-1, 2).float()

        loss = self.criterion(output_flat, y)

        self.val_metric.update({k:v.cpu() for k,v in batch.items()}, output.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)
        return loss


class TrajectoryPredictorTFT(TrajectoryPredictor):
    def __init__(self, config, scalers_dict):
        data_props = {
            'num_historical_numeric': 14, 
            'num_static_categorical': 6,
            # vehicle_model, vehicle_model_modification, location_reference_point_id, tires_front, tires_rear, ride_month
            'static_categorical_cardinalities': [2, 6, 3, 15, 15, 12],
            'num_future_numeric': 2,
        }

        configuration = Dict({
            'model':
                {
                    'dropout': 0.1,
                    'state_size': 64,
                    'output_quantiles': [0.5],
                    'lstm_layers': 2,
                    'attention_heads': 4
                },
            'task_type': 'regression',
            'target_window_start': None,
            'data_props': data_props
        })

        config.model.configuration = configuration

        super().__init__(config, scalers_dict)

    def forward(self, batch):
        tft_batch = {
            'historical_ts_numeric': batch["history"].float(),
            'future_ts_numeric': batch["future"].float(),
            'static_feats_categorical': batch["static"].long(),
        }

        output = self.model(tft_batch)
        output = output["predicted_quantiles"]

        return output


class TrajectoryPredictorTFTYaw(TrajectoryPredictor):
    def __init__(self, config, scalers_dict):
        data_props = {
            'num_historical_numeric': 14, 
            'num_static_categorical': 6,
            # vehicle_model, vehicle_model_modification, location_reference_point_id, tires_front, tires_rear, ride_month
            'static_categorical_cardinalities': [2, 6, 3, 15, 15, 12],
            'num_future_numeric': 2,
        }

        configuration = Dict({
            'model':
                {
                    'dropout': 0.1,
                    'state_size': 64,
                    'output_quantiles': [0.5, 0.5],
                    'lstm_layers': 2,
                    'attention_heads': 4
                },
            'task_type': 'regression',
            'target_window_start': None,
            'data_props': data_props
        })

        config.model.configuration = configuration

        super().__init__(config, scalers_dict)

    def forward(self, batch):
        tft_batch = {
            'historical_ts_numeric': batch["history"].float(),
            'future_ts_numeric': batch["future"].float(),
            'static_feats_categorical': batch["static"].long(),
        }

        output = self.model(tft_batch)
        output = output["predicted_quantiles"]

        output = F.normalize(output, p=2.0, dim=2)

        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        output_flat = output.reshape(-1, 2)
        y = batch["y"].reshape(-1, 2).float()

        loss = self.criterion(output_flat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        output_flat = output.reshape(-1, 2)
        y = batch["y"].reshape(-1, 2).float()

        loss = self.criterion(output_flat, y)

        self.val_metric.update({k:v.cpu() for k,v in batch.items()}, output.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)
        return loss


class TrajectoryPredictorTFTQuantile(TrajectoryPredictor):
    def __init__(self, config, scalers_dict):
        data_props = {
            'num_historical_numeric': 14, 
            'num_static_categorical': 6,
            # vehicle_model, vehicle_model_modification, location_reference_point_id, tires_front, tires_rear, ride_month
            'static_categorical_cardinalities': [2, 6, 3, 15, 15, 12],
            'num_future_numeric': 2,
        }

        configuration = Dict({
            'model':
                {
                    'dropout': 0.1,
                    'state_size': 64,
                    'output_quantiles': [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
                    'lstm_layers': 2,
                    'attention_heads': 4
                },
            'task_type': 'regression',
            'target_window_start': None,
            'data_props': data_props
        })

        config.model.configuration = configuration

        super().__init__(config, scalers_dict)

    def forward(self, batch):
        tft_batch = {
            'historical_ts_numeric': batch["history"].float(),
            'future_ts_numeric': batch["future"].float(),
            'static_feats_categorical': batch["static"].long(),
        }

        output = self.model(tft_batch)
        output = output["predicted_quantiles"]
        output, _ = output.sort(-1)

        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = self.criterion(output, batch["y"].squeeze().float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        loss = self.criterion(output, batch["y"].squeeze().float())

        # predictions = self.criterion.to_prediction(output)
        predictions = output[..., 3]

        if self.val_metric:
            self.val_metric.update({k:v.cpu() for k,v in batch.items()}, predictions.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)
        return loss