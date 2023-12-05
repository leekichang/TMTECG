"""
Created on Thu Aug 18 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""

import torch
import utils
from tqdm import tqdm

def load_backbone(ckpt, model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = utils.build_trainer(args)
    checkpoint = torch.load(f'./checkpoints/OURS_linear_whole/100.pth')
    
    load_backbone(checkpoint, trainer.model)
    # for param_name, param in trainer.model.named_parameters():
    #     if 'classifier' not in param_name:
    #         param.requires_grad = False
            
    trainer.test()
    trainer.print_train_info()