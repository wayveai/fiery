import torch
import torch.nn as nn

from fiery.layers.convolutions import Bottleneck


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.min_std = 0.1

        self.encoder = DistributionEncoder(
            in_channels,
            self.compress_dim,
        )
        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )

    def forward(self, s_t):
        def sigmoid2(tensor, min_value):
            return 2 * torch.sigmoid(tensor / 2) + min_value

        b, s = s_t.shape[:2]
        assert s == 1
        encoding = self.encoder(s_t[:, 0])

        mu_log_sigma = self.last_conv(encoding).view(b, 1, 2 * self.latent_dim)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class DistributionEncoder(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            Bottleneck(in_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
        )

    def forward(self, s_t):
        return self.model(s_t)
