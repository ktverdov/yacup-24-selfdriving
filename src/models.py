import torch
import torch.nn as nn
from torchtsmixer import TSMixerExt

from omegaconf import OmegaConf
from tft_torch import tft


def get_model(model_config, tft_configuration=None):
    if model_config.type == "transformer":
        model = TransformerModel(**model_config.parameters)
    elif model_config.type == "tsmixer_ext":
        print(model_config.parameters)
        model = TSMixerExt(**model_config.parameters)
    elif model_config.type == "tft":
        print(model_config)
        model = tft.TemporalFusionTransformer(model_config.configuration)
    else:
        return None

    return model


class TransformerModel(nn.Module):
    def __init__(self, history_dim, future_dim, output_dim, 
                 nhead, d_model, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        self.history_emb = nn.Linear(history_dim, d_model)
        self.future_emb = nn.Linear(future_dim, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, output_dim)
        # self.static_fc = nn.Linear(static_dim, output_dim)

    def forward(self, history_input, future_input, static_input):
        encoder_input = self.history_emb(history_input)
        decoder_input = self.future_emb(future_input)

        transformer_output = self.transformer(encoder_input, decoder_input)

        output = self.fc(transformer_output)
        return output
