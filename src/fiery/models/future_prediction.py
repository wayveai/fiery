import torch

from fiery.layers.convolutions import Bottleneck
from fiery.layers.temporal import SpatialGRU


class FuturePrediction(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.n_gru_blocks = n_gru_blocks

        # Convolutional recurrent model with z_t as an initial hidden state and inputs the sample
        # from the probabilistic model. The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(torch.nn.Sequential(*[Bottleneck(in_channels)
                                                         for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, x, hidden_state):
        # x has shape (b, n_future, c, h, w), hidden_state (b, c, h, w)
        for i in range(self.n_gru_blocks):
            x = self.spatial_grus[i](x, hidden_state, flow=None)
            b, n_future, c, h, w = x.shape

            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)

        return x
