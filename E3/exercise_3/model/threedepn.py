import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80
        kernel_size = 4 
        stride = 2

        self.enc1 = nn.Sequential(
            nn.Conv3d(
                2, 
                self.num_features, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(
                self.num_features, 
                2 * self.num_features, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=1),
            nn.BatchNorm3d(2 * self.num_features),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(
                2 * self.num_features, 
                4 * self.num_features, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1),
            nn.BatchNorm3d(4 * self.num_features),
            nn.LeakyReLU(0.2)
        )        
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(
                4 * self.num_features, 
                8 * self.num_features, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=0),
            nn.BatchNorm3d(8 * self.num_features),
            nn.LeakyReLU(0.2)
        )        

        # TODO: 2 Bottleneck layers
        
        self.bottleneck = nn.Sequential(
            nn.Linear(8 * self.num_features, 8 * self.num_features), 
            nn.ReLU(inplace=True),
            nn.Linear(8 * self.num_features, 8 * self.num_features), 
            nn.ReLU(inplace=True),
        )
        
        # TODO: 4 Decoder layers
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(
                2 * 8 * self.num_features, 
                4 * self.num_features, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=0),
            nn.BatchNorm3d(4 * self.num_features),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(
                2 * 4 * self.num_features, 
                2 * self.num_features, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=1),
            nn.BatchNorm3d(2 * self.num_features),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(
                2 * 2 * self.num_features, 
                self.num_features, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU()
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(
                2 * self.num_features, 
                1, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=1)
        )        
        
        

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # Reshape and apply bottleneck layers
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # print(e1.size(), e2.size(), e3.size(), e4.size())

        x = e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        
        
        # Decode
        
        x = torch.cat([x, e4], dim=1)
        x = self.dec1(x)
       
        x = torch.cat([x, e3], dim=1)
        x = self.dec2(x)
        
        x = torch.cat([x, e2], dim=1)
        x = self.dec3(x)
        
        x = torch.cat([x, e1], dim=1)
        x = self.dec4(x)

        x = torch.squeeze(x, dim=1)
        
        x = torch.log(torch.abs(x) + 1)

        return x
