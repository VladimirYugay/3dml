import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        self.l1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(latent_size + 3, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512 - (latent_size + 3))),
            nn.ReLU(),
        )
        
        self.l2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 1)),
        )

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        x = self.l1(x_in)
        x = torch.cat([x, x_in], axis=-1)
        x = self.l2(x)
        return x
