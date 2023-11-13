import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        identity = x.clone()
        x = self.conv2((self.conv1(x))) #conv -> dropout -> conv
        x = self.relu(x)
        x = self.bn2(x) # BN lecture: bn after relu is okay, if before use parameters

        x += identity

        return x



class RNN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=5,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.res_block_1 = nn.Sequential(
        			        BasicBlock(in_channels=hidden_units*2,
                                       out_channels=hidden_units*2, stride=1, is_first_block=False),
                            BasicBlock(in_channels=hidden_units*2,
                                       out_channels=hidden_units*2, stride=1, is_first_block=False),
                            BasicBlock(in_channels=hidden_units*2,
                                       out_channels=hidden_units*2, stride=1, is_first_block=False),
                                       )

        self.res_block_2 = nn.Sequential(
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
        )

        self.dropout = nn.Dropout2d(p=0.02)

        self.res_block_3 = nn.Sequential(
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
            BasicBlock(in_channels=hidden_units * 2,
                       out_channels=hidden_units * 2, stride=1, is_first_block=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*4,
                      out_features=64),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=64,
                      out_features=output_shape))

    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"After 1. block: {x.shape}")
        x = self.res_block_1(x)
        #print(f"After res block: {x.shape}")
        x = self.res_block_2(x)
        #print(f"After res block: {x.shape}")
        x = self.res_block_3(x)
        #print(f"After res block: {x.shape}")
        x = self.avgpool(x)
        #print(f"After avgpool: {x.shape}")
        x = self.classifier(x)
        #print(f"After classifier: {x.shape}")
        return x
