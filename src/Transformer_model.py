# model.py

import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, num_variates, lookback_len, pred_length, dim, depth):
        super(TransformerModel, self).__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        self.dim = dim

        self.input_layer = nn.Linear(num_variates, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_layer = nn.Linear(dim, num_variates)

    def forward(self, src):
        # src: (batch_size, lookback_len, num_variates)
        src = self.input_layer(src)  # (batch_size, lookback_len, dim)
        src = src.permute(1, 0, 2)   # (lookback_len, batch_size, dim)
        output = self.transformer_encoder(src)  # (lookback_len, batch_size, dim)
        output = output.permute(1, 0, 2)  # (batch_size, lookback_len, dim)
        output = self.output_layer(output[:, -self.pred_length:, :])  # (batch_size, pred_length, num_variates)
        return output


class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = (-val_loss)
        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score == self.best_score:#score < self.best_score:
            """self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True"""
            self.early_stop = True #付けたし
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
    

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        # Save the model (you might want to implement this to actually save the model)
