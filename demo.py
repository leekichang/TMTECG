import torch
import utils
import numpy as np
from torch.utils.data import DataLoader

import sklearn.metrics as metrics

if __name__ == '__main__':
    args  = utils.parse_args()
    stage = f'STAGE{args.stage}'
    model = utils.build_model(args).to('cuda')
    
    model.load_state_dict(torch.load('./checkpoints/weighted/50.pth'))
    dataset = utils.load_dataset(args, is_train=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle = False, drop_last=False)
    
    with torch.no_grad():
        model.eval()
        preds, targets, losses = [], [], []
        for X, Y in dataloader:
            X, Y = X.to('cuda'), Y.to('cuda')
            pred = model(X)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            
        preds   = np.concatenate(preds)
        targets = np.concatenate(targets)        
        acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        print(acc)
        result  = metrics.classification_report(y_true=targets, y_pred=np.argmax(preds, axis=-1))
        print(result)
        print(f'specificity_score: {utils.specificity_score(y_true=targets, y_pred=np.argmax(preds, axis=-1)):.4f}')
        confmap = metrics.confusion_matrix(y_true=targets, y_pred=np.argmax(preds, axis=-1))
        print(confmap)
        