"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch
import utils
from tqdm import tqdm

if __name__ == '__main__':
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = utils.build_trainer(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0:
            trainer.save_model()
        trainer.epoch += 1