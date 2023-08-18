"""
Created on Thu Aug 18 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""

import torch
import utils
from tqdm import tqdm

def load_backbone(ckpt, learner):
    for key, value in ckpt.items():
        learner.model.state_dict()[key] = value

if __name__ == '__main__':
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = utils.build_trainer(args)
    checkpoint = torch.load('./checkpoints/BYOL_2048/10.pth')

    load_backbone(checkpoint, trainer)
    for param_name, param in trainer.model.named_parameters():
        if 'classifier' not in param_name:
            param.requires_grad = False
            
            
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0:
            trainer.save_model()
        trainer.epoch += 1