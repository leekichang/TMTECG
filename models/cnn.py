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
                'groups' :1,
                'n_stage':6,},
                
                'single-B':{'channel':[12, 4, 16, 32],
                'kernel' : 7,
                'stride' : 3,
                'linear' :[320,256],
                'groups' :1,
                'n_stage':1,},
            
                'Bg':{'channel':[12*1, 12*4, 12*16, 12*32],
                'kernel' : 7,
                'stride' : 3,
                'linear' :[320,256],
                'groups' :12,
                'n_stage':6,}
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
    
def MLP(dim, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, dim)
    )

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
        self.projector   = MLP(dim=self.config['linear'][1]*self.config['n_stage'], hidden_size=512*self.config['n_stage'])
        if num_class != 0:
            self.classifier  = nn.Linear(in_features=self.config['linear'][1]*self.config['n_stage'], out_features=num_class)
        else:
            self.classifier  = nn.Identity()
    
    def forward(self, x):
        B = x.size(0)
        if len(x.shape) == 4:
            x = x.reshape((x.size(0)*self.config['n_stage'], x.size(2), x.size(3)))
        for idx in range(len(self.convBlocks)):
            x = self.drop_outs[idx](self.convBlocks[idx](x))
        x = self.head(x.reshape(B, self.config['n_stage'], -1))
        x = self.classifier(x.reshape(x.size(0), -1))
        return x
    

if __name__ == '__main__':
    model = CNN('B')
    x = torch.randn(2, 6, 12, 2500)
    print(0, x.shape)
    out = model(x)
    print(out.shape)