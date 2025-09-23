import torch 
from torch import nn 
class ConvProjector(nn.Module):
    def __init__(self, in_channels=576, in_h=32, in_w=32, out_h=1374, out_w=2048, mid_channels=512,out_channels=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        # 用1x1卷积降维通道到1，方便输出二维张量
        self.channel_reducer = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        print(f"importing conv projector from {__file__}")
    def forward(self, x):
        """
        x: (batch, 576, 32, 32)
        output: (batch,24, 1374, 2048) 
        """
        x = self.conv(x)             # (batch, mid_channels, 32, 32)
        x = self.pool(x)             # (batch, mid_channels, 1374, 2048)
        x = self.channel_reducer(x)  # (batch, 1, 1374, 2048)
        x = x.squeeze(1)             # (batch,24, 1374, 2048)
        return x
