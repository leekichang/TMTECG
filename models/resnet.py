import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResNet"
]

model_config={  
                'B':{
                'n_stage':6,},
                
                'single-B':{
                'n_stage':1,},
          }

class resConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer_num):
        super(resConv1dBlock, self).__init__()
        self.layer_num = layer_num
        self.conv1 = nn.ModuleList([
            nn.Conv1d(in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn1 = nn.ModuleList([
            nn.BatchNorm1d(2 * in_channels)
            for i in range(layer_num)])

        self.conv2 = nn.ModuleList([ 
            nn.Conv1d(in_channels = 2 * in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn2 = nn.ModuleList([
            nn.BatchNorm1d(out_channels)
            for i in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            tmp = F.relu(self.bn1[i](self.conv1[i](x)))
            x = F.relu(self.bn2[i](self.conv2[i](tmp)) + x)
        return x

def MLP(dim, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, dim)
    )

class ResNet(nn.Module):
    def __init__(self, model_type='single-B', num_class=2):
        super(ResNet, self).__init__()
        self.config = model_config[model_type]
        input_channel, input_size = 12, 2500
        
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 1, stride = 1)
        self.res1 = resConv1dBlock(64, 64, kernel_size = 3, stride = 1, layer_num = 1)
        self.pool1 = nn.AvgPool1d(kernel_size = 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        self.res2 = resConv1dBlock(128, 128, kernel_size = 3, stride = 1, layer_num = 1)
        self.pool2 = nn.AvgPool1d(kernel_size = 2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size = 1, stride = 1)
        self.res3 = resConv1dBlock(256, 256,  kernel_size = 3, stride = 1, layer_num = 1)
        self.pool3 = nn.AvgPool1d(kernel_size = 2)

        self.conv4 = nn.Conv1d(256, 128, kernel_size = 1, stride = 1)
        self.res4 = resConv1dBlock(128, 128, kernel_size = 3, stride = 1, layer_num = 1)
        self.pool = nn.AvgPool1d(kernel_size = int(input_size / 8))
        
        self.projector   = MLP(dim=128, hidden_size=256*self.config['n_stage'])
        
        if num_class != 0:
            self.classifier  = nn.Linear(in_features=128*self.config['n_stage'], out_features=num_class)
        else:
            self.classifier  = nn.Identity()

    def forward(self, x):
        B = x.size(0)
        if len(x.shape) == 4:
            x = x.reshape((x.size(0)*self.config['n_stage'], x.size(2), x.size(3)))
        # print(f'0:{x.shape}')
        x = F.relu(self.conv1(x))
        # print(f'1: {x.shape}')
        x = self.pool1(self.res1(x))
        # print(f'2: {x.shape}')
        x = F.relu(self.conv2(x))
        # print(f'3: {x.shape}')
        x = self.pool2(self.res2(x))
        # print(f'4: {x.shape}')
        x = F.relu(self.conv3(x))
        # print(f'5: {x.shape}')
        x = self.pool3(self.res3(x))
        # print(f'6: {x.shape}')
        x = F.relu(self.conv4(x))
        # print(f'7: {x.shape}')
        x = self.pool(self.res4(x))
        # print(f'8: {x.shape}')
        x = x.reshape(B, self.config['n_stage'], -1)
        # print(f'8.5: {x.shape}')
        x = x.reshape(B, -1)
        # print(f'9: {x.shape}')
        x = self.classifier(x)
        # print(f'10: {x.shape}')
        return x

def main():
    input = torch.zeros((2, 12, 2500)).cuda()
    model = ResNet(model_type='single-B', num_class = 2).cuda()
    model.projector = None
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f'Total parameter:{total_params:,}')    
    o = model(input)
    print(o.size())

if __name__ == '__main__':
	main()

# import os
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch

# __all__ = [
#     'resnet'
# ]

# def _padding(downsample, kernel_size):
#     """Compute required padding"""
#     padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
#     return padding


# def _downsample(n_samples_in, n_samples_out):
#     """Compute downsample rate"""
#     downsample = int(n_samples_in // n_samples_out)
#     if downsample < 1:
#         raise ValueError("Number of samples should always decrease")
#     if n_samples_in % n_samples_out != 0:
#         raise ValueError("Number of samples for two consecutive blocks "
#                          "should always decrease by an integer factor.")
#     return downsample

# class ResBlock1d(nn.Module):
#     """Residual network unit for unidimensional signals."""

#     def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
#         if kernel_size % 2 == 0:
#             raise ValueError("The current implementation only support odd values for `kernel_size`.")
#         super(ResBlock1d, self).__init__()
#         # Forward path
#         padding = _padding(1, kernel_size)
#         self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
#         self.bn1 = nn.BatchNorm1d(n_filters_out)
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout_rate)
#         padding = _padding(downsample, kernel_size)
#         self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,stride=downsample, padding=padding, bias=False)
#         self.bn2 = nn.BatchNorm1d(n_filters_out)
#         self.dropout2 = nn.Dropout(dropout_rate)

#         # Skip connection
#         skip_connection_layers = []
#         # Deal with downsampling
#         if downsample > 1:
#             maxpool = nn.MaxPool1d(downsample, stride=downsample)
#             skip_connection_layers += [maxpool]
#         # Deal with n_filters dimension increase
#         if n_filters_in != n_filters_out:
#             conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
#             skip_connection_layers += [conv1x1]
#         # Build skip conection layer
#         if skip_connection_layers:
#             self.skip_connection = nn.Sequential(*skip_connection_layers)
#         else:
#             self.skip_connection = None

#     def forward(self, x, y):
#         """Residual unit."""
#         if self.skip_connection is not None:
#             y = self.skip_connection(y)
#         else:
#             y = y
#         # 1st layer
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout1(x)

#         # 2nd layer
#         x = self.conv2(x)
#         x += y  # Sum skip connection and main connection
#         y = x
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout2(x)
#         return x, y
        
# class resnet(nn.Module):
#     """Residual network for unidimensional signals.
#     Parameters
#     ----------
#     input_dim : tuple
#         Input dimensions. Tuple containing dimensions for the neural network
#         input tensor. Should be like: ``(n_filters, n_samples)``.
#     blocks_dim : list of tuples
#         Dimensions of residual blocks.  The i-th tuple should contain the dimensions
#         of the output (i-1)-th residual block and the input to the i-th residual
#         block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
#         for two consecutive samples should always decrease by an integer factor.
#     n_classes : classes to predict
#         Predicting single value; age. Default 1.  
#     dropout_rate: float [0, 1), optional
#         Dropout rate used in all Dropout layers. Default is 0.5. 
#     kernel_size: int, optional
#         Kernel size for convolutional layers. The current implementation
#         only supports odd kernel sizes. Default is 17.
#     References
#     ----------
#     .. [1] Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a
#            mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
#     .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
#            arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
#     .. [3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
#            on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
#     """

#     def __init__(self, model_type='', num_class=2):
#         super(resnet, self).__init__()
#         input_dim   = (12,2500)
        
#         filters     = [24, 36, 48, 64]
#         seq_lengths = [2500, 625, 125, 25]
#         blocks_dim  = list(zip(filters, seq_lengths))
        
#         kernel_size=17
#         dropout_rate=0.8
#         # First layers
#         n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
#         n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
#         downsample = _downsample(n_samples_in, n_samples_out)
#         padding = _padding(downsample, kernel_size)
#         self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
#                                stride=downsample, padding=padding)
#         self.bn1 = nn.BatchNorm1d(n_filters_out)
#         self.relu = nn.ReLU()

#         # Residual block layers
#         self.res_blocks = nn.ModuleList()
#         for i, (n_filters, n_samples) in enumerate(blocks_dim):
#             n_filters_in, n_filters_out = n_filters_out, n_filters
#             n_samples_in, n_samples_out = n_samples_out, n_samples
#             downsample = _downsample(n_samples_in, n_samples_out)
#             resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
#             self.res_blocks += [resblk1d]

#         # Linear layer
#         n_filters_last, n_samples_last = blocks_dim[-1]
#         last_layer_dim = n_filters_last * n_samples_last
#         if num_class == 0:
#             self.lin = nn.Identity()
#         else:
#             self.lin = nn.Linear(last_layer_dim, num_class)
#         self.n_blk = len(blocks_dim)

#     def forward(self, x):
#         """Implement ResNet1d forward propagation"""
#         # First layers
#         x = self.conv1(x)
#         x = self.bn1(x)

#         # Residual blocks
#         y = x
#         for blk in self.res_blocks:
#             x, y = blk(x, y)

#         # Flatten array
#         x = x.view(x.size(0), -1)

#         # Fully conected layer
#         x = self.lin(x)
#         return x
    
# if __name__ == '__main__':
    
#     model = resnet(num_class=0)
#     x = torch.rand(40,12,2500)
#     out = model(x)
#     print(x.shape, out.shape)