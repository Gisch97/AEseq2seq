import torch
from torch import nn
### CONVOLUTION 


# Definición de N_Conv: Una secuencia de Conv1d -> BatchNorm1d -> ReLU
class N_Conv(nn.Module):
    '''([Conv] -> [BatchNorm] -> [ReLu]) x N'''
    
    def __init__(self, input_channels, output_channels, num_conv, kernel_size=3, padding=1 , stride=1, AVG_POOL=False):
        super().__init__()
        layers = []
        for i in range(num_conv):
            if AVG_POOL:
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2, padding=0))  
    
            if i != 0:          
                layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)) 
            else:
                layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride))
            layers.append(nn.BatchNorm1d(output_channels)) 
            layers.append(nn.ReLU(inplace=True))
    
        self.N_Conv = nn.Sequential(*layers)    
        
    def forward(self, x):
        return self.N_Conv(x)
# Definición de N_Conv: Una secuencia de Conv1d -> BatchNorm1d -> ReLU

class Max_Down(nn.Module):
    """Downscaling with maxpool then N_Conv"""
    def __init__(self, in_channels, out_channels, num_conv, kernel_size=3, padding=1 , stride=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            N_Conv(in_channels, out_channels, num_conv, kernel_size=kernel_size, padding=padding, stride=stride)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    

class Avg_Down(nn.Module):
    """Downscaling with avgpool then N_Conv"""
    def __init__(self, in_channels, out_channels, num_conv, kernel_size=3, padding=1 , stride=1):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
            N_Conv(in_channels, out_channels, num_conv, kernel_size, padding, stride)
        )
    def forward(self, x):
        return self.avgpool_conv(x)
    
    
    
    
class Up_Block(nn.Module):
    """Upscaling then N conv"""

    def __init__(self, in_channels, out_channels, num_conv, up_mode= 'upsample', addition='cat', skip=True, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.skip = skip
        self.addition = addition
        
        if up_mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        elif up_mode == 'traspose':
            self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            raise ValueError('up_mode must be "upsample" or "traspose"')

        if skip:
            if addition == 'cat':
                conv_in_channels = in_channels + out_channels 
            elif addition == 'sum':
                conv_in_channels = out_channels
                self.adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            else:
                raise ValueError('Addition must be "cat" or "sum"')
 
        self.conv = N_Conv(conv_in_channels, out_channels, num_conv, kernel_size, padding, stride)
    def forward(self, x1, x2):
        print(f'#####    X1    #########    {x1.shape}    ##########    X1    ########## X1     ##########')
        x1 = self.up(x1)
        
        print(f'#####    UP    #########    {x1.shape}    ##########    UP    ########## X1 UP    ##########')
        if self.skip:
            if self.addition == 'cat':
                diff = x2.size()[2] - x1.size()[2]
                x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2]) 
                x = torch.cat([x2, x1], dim=1)  
            elif self.addition == 'sum': 
                x1 = self.adjust(x1) 
                x = x2 + x1
            else:
                raise ValueError('Addition must be "cat" or "sum"')
        else:
            x = x1
        return self.conv(x)
    
    


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)