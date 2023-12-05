"""
Created on Thu Sep 06 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch.nn as nn

__all__ = ['vit']

model_config={  
                'B':{'n_stage':6,
                     'linear':[6400, 2048, 512],
                     'hidden':512                     
                     },
                
                'B-single':{'n_stage':1},
          }

class Rearrange(nn.Module):
    def __init__(self, dim1=1, dim2=2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class TokenEmbedding(nn.Module):
    def __init__(self, patch_size=100, hidden_dim=512):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size), # B, 12, 2500 -> B, 128, 50
            Rearrange(1,2) # B, 128, 50 -> B, 50, 128
        )
        
    def forward(self, x):
        return self.embedding(x)

def MLP(dim, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, dim)
    )

class vit(nn.Module):
    def __init__(self, model_type='B', input_size=2500, num_class=2, patch_size=50, hidden_dim=128, num_heads=4, num_layers=4):
        super(vit, self).__init__()
        self.config = model_config[model_type]
        assert input_size % patch_size == 0, "Input dimensions must be divisible by the patch size."
        self.patch_size      = patch_size
        self.num_patches     = input_size // patch_size
        self.token_embedding = TokenEmbedding(patch_size=patch_size, hidden_dim=hidden_dim)
        encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=256,)
        self.transformer = nn.TransformerEncoder(encoder, num_layers)
        
        self.head = nn.Sequential(
                    nn.Linear(self.config['linear'][0], self.config['linear'][1]),
                    nn.Linear(self.config['linear'][1], self.config['linear'][2])
                    )
        
        self.projector  = MLP(dim=self.config['linear'][2]*self.config['n_stage'], hidden_size=self.config['hidden']*self.config['n_stage'])
        if num_class != 0:
            self.classifier  = nn.Linear(in_features=self.config['linear'][2]*self.config['n_stage'], out_features=num_class)
        else:
            self.classifier  = nn.Identity()

    def forward(self, x):
        B = x.size(0)
        # print(1, f'{x.shape}')
        if len(x.shape) == 4:
            x = x.reshape((x.size(0)*self.config['n_stage'], x.size(2), x.size(3)))
        # print(2, f'{x.shape}')
        x = self.token_embedding(x)
        # print(3, f'{x.shape}')
        x = self.transformer(x)
        # print(4, f'{x.shape}')
        x = x.reshape(B, self.config['n_stage'], -1)
        # print(5, f'{x.shape}')
        x = self.head(x)
        # print(6, f'{x.shape}')
        x = x.reshape(B, -1)
        # print(7, f'{x.shape}')
        x = self.classifier(x)
        # print(8, f'{x.shape}')

        return x
    
if __name__ == '__main__':
    import torch
    # from torchsummary import summary
    model = vit(num_class=2).to('cuda')
    print(model)
    # summary(model, (3,32,32))
    x = torch.rand(2,6,12,2500).to('cuda')
    out = model(x)
    print(x.shape, out.shape)