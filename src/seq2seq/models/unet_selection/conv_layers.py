from torch import nn

### CONVOLUTION 
def conv3_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride))
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True))
    
    for _ in range(num_conv - 1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)) 
        layers.append(nn.BatchNorm1d(output_channels)) 
        layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def tconv3_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
        layers.append(nn.BatchNorm1d(input_channels)) 
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)


def conv2_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride))
    layers.append(nn.ReLU(inplace=True))       
    for _ in range(num_conv - 1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)) 
        layers.append(nn.ReLU(inplace=True))    

    return nn.Sequential(*layers)

def tconv2_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
        layers.append(nn.ReLU(inplace=True))    
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
    layers.append(nn.ReLU(inplace=True))    

    return nn.Sequential(*layers)



def conv1_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)) 
    for _ in range(num_conv - 1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride))  
    return nn.Sequential(*layers)

def tconv1_c(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1))   
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1))  
    return nn.Sequential(*layers)
    
    
### AVG POOLING
def conv3_ap(input_channels, output_channels, num_conv=1,  padding=1, pool_stride=2, pool_kernel=2): 
    
    layers = [] 
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=padding, stride=1))
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True)) 
    layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0))  
    for _ in range(num_conv-1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=padding, stride=1))    
        layers.append(nn.BatchNorm1d(output_channels))
        layers.append(nn.ReLU(inplace=True))     
        layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    return nn.Sequential(*layers)


def tconv3_ap(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
        layers.append(nn.BatchNorm1d(input_channels)) 
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True)) 
    return nn.Sequential(*layers)


def conv1_ap(input_channels, output_channels, num_conv=1,  padding=1, pool_stride=2, pool_kernel=2): 
    
    layers = [] 
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=padding, stride=1)) 
    layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    
    for _ in range(num_conv-1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=padding, stride=1))     
        layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    return nn.Sequential(*layers)


def tconv1_ap(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1))  
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1))  
    return nn.Sequential(*layers)

def conv2_ap(input_channels, output_channels, num_conv=1,  padding=1, pool_stride=2, pool_kernel=2): 
    
    layers = [] 
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=padding, stride=1))
    layers.append(nn.ReLU(inplace=True))    

    layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    
    for _ in range(num_conv-1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=padding, stride=1))    
        layers.append(nn.ReLU(inplace=True))    
        layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    return nn.Sequential(*layers)


def tconv2_ap(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
        layers.append(nn.ReLU(inplace=True))    
  
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1))  
    layers.append(nn.ReLU(inplace=True))    
    return nn.Sequential(*layers)
