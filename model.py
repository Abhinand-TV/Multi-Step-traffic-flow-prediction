import torch
import torch.nn as nn


class TrafficTransformer(nn.Module):

    def __init__(self, input_dim, d_model=32, n_heads=2, num_layers=1, pred_len=2):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 12, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):

        x = self.embed(x)
        x = x + self.pos

        x = self.transformer(x)

        x = x[:, -1, :]

        return self.fc(x)