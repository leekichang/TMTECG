import torch
import utils
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

import matplotlib.pyplot as plt


if __name__ == '__main__':
    args  = utils.parse_args()
    stage = f'STAGE{args.stage}'
    model = utils.build_model(args).to('cuda')
    # model.load_state_dict(torch.load(f'./checkpoints/baseline_all/100.pth'))
    # model.load_state_dict(torch.load(f'./checkpoints/SimCLR_{args.test_batch}_{args.ckpt_epoch}_1e-4_{args.stage}/100.pth'))
    model.load_state_dict(torch.load(f'./checkpoints/BYOL_{args.test_batch}_{args.ckpt_epoch}_1e-4_{args.stage}/80.pth'))
    #model.load_state_dict(torch.load(f'./checkpoints/baseline_{args.stage}/100.pth'))
    args.dataset = 'TMT'
    # args.is_whole = True
    
    dataset = utils.load_dataset(args, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle = False, drop_last=False)
    
    with torch.no_grad():
        model.eval()
        probs, targets, losses = [], [], []
        for X, Y in dataloader:
            X, Y = X.to('cuda'), Y.to('cuda')
            pred = model(X)
            prob = torch.softmax(pred, dim=1)
            
            probs.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
        
        probs   = np.concatenate(probs)
        targets = np.concatenate(targets)        
        
        acc     = utils.calculate_topk_accuracy(torch.from_numpy(probs), torch.from_numpy(targets), k=1)
        recall  = recall_score(y_true=targets, y_pred=np.argmax(probs, axis=-1))
        f1      = f1_score(y_true=targets, y_pred=np.argmax(probs, axis=-1))
        spec    = utils.specificity_score(y_true=targets, y_pred=np.argmax(probs, axis=-1))
        bal_acc = balanced_accuracy_score(y_true=targets, y_pred=np.argmax(probs, axis=-1))
        auroc   = roc_auc_score(y_true=targets, y_score=probs[:, 1])
        
        tn, fp, fn, tp = confusion_matrix(targets, np.argmax(probs, axis=-1)).ravel()
        sensitivity = recall
        specificity = spec
        youden_index = sensitivity + specificity - 1
        
        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
        sensitivity = tpr
        specificity = 1 - fpr
        j_index = sensitivity + specificity - 1
        best_threshold_idx = np.argmax(j_index)
        best_threshold = thresholds[best_threshold_idx]
        binary_predictions = (probs[:, 1] > best_threshold).astype(int)
        
        acc     = accuracy_score(y_true=targets, y_pred=binary_predictions)
        recall  = recall_score(y_true=targets, y_pred=binary_predictions)
        f1      = f1_score(y_true=targets, y_pred=binary_predictions)
        spec    = utils.specificity_score(y_true=targets, y_pred=binary_predictions)
        bal_acc = balanced_accuracy_score(y_true=targets, y_pred=binary_predictions)
        
        print(f'Test Accuracy:{acc:>5.2f}% Balanced Test Accuracy:{bal_acc:>5.2f}% sensitivity:{recall:>6.4f} specification:{spec:>5.4f} auroc:{auroc:>5.4f}', flush=True)
        
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig('./ROC.png')
        plt.show()