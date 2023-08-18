"""
Created on Thu Aug 16 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch
import torch.nn as nn

__all__ = [
    'CNN',
    ]

model_config={  
                'B':{'channel':[12, 4, 16, 32],
                'kernel' : 7,
                'stride' : 3,
                'linear' :[320,256],
                'groups' :1},
            
                'Bg':{'channel':[12*1, 12*4, 12*16, 12*32],
                'kernel' : 7,
                'stride' : 3,
                'linear' :[320,256],
                'groups' :12}
          }

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups=1):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, groups=groups),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    
    def forward(self, x):
        x = self.convs(x)
        return x

class DummyLayer(nn.Module):
    def __init__(self, groups):
        self.groups = groups
        pass
    def forward(self, x):
        return x.reshape(x.size(0), self.groups, -1)

class CNN(nn.Module):
    def __init__(self, model_type='Bg', num_class=2):
        super(CNN, self).__init__()
        self.config     = model_config[model_type]
        self.convBlocks = nn.ModuleList([])
        for i in range(len(self.config['channel'])-1):
            self.convBlocks.append(ConvBlock(in_channel=self.config['channel'][i],
                                             out_channel=self.config['channel'][i+1],
                                             kernel_size=self.config['kernel'],
                                             stride=self.config['stride'],
                                             groups=self.config['groups']))
            
        self.drop_outs   = nn.ModuleList([nn.Dropout(0.1) for _ in range(3)])
        
        self.head        = nn.Linear(in_features=self.config['linear'][0], out_features=self.config['linear'][1])
        if num_class != 0:
            self.classifier  = nn.Linear(in_features=self.config['linear'][1]*self.config['groups'], out_features=num_class)
        else:
            self.classifier  = DummyLayer(self.config['groups'])
            
    
    def forward(self, x):
        for idx in range(len(self.convBlocks)):
            x = self.drop_outs[idx](self.convBlocks[idx](x))
        x = self.head(x.reshape(x.size(0), self.config['groups'], -1))
        x = self.classifier(x.reshape(x.size(0), -1))
        return x
    

if __name__ == '__main__':
    model = CNN('B')
    x = torch.randn(1, 12, 2500)
    print(x.shape)
    out = model(x)
    print(out.shape)