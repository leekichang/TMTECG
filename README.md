# TMTECG
TreadMill Test ECG


## Pretraining

For model pretraining, this repository four contrastive learning based pretraining methods (i.e., BYOL, SimCLR, CMSC, OURS). The trained models are saved in `./checkpoints/{exp_name}`.

### BYOL

```
python BYOL.py --phase BYOL --dataset full --trainset full --batch_size 4096 --model CNN_single-B --use_tb T --exp_name BYOL_pretrain
```

### SimCLR
```
python SimCLR.py --phase SimCLR --dataset full --trainset full --batch_size 4096 --model CNN_single-B --use_tb T --exp_name SimCLR_pretrain
```

### CMSC
```
python CMSC.py --phase CMSC --dataset full --trainset full --batch_size 4096 --model CNN_single-B --use_tb T --exp_name CMSC_pretrain
```

### OURS
```
python OURS.py --phase OURS --dataset full --trainset full --batch_size 4096 --model CNN_single-B --use_tb T --exp_name OURS_pretrain
```


## Model Evaluation
For model evaluation, this repository supports `finetuning` and `linear-probing`.

`finetuning` trains the whole model including pretrained backbone.

`linear-probing` only trains the classifier while the pretrained backbone is freezed.

You can select option for `dataset` between `angio`, `cad`, `whole`.

`angio` dataset include subjects who have angiogram test results.

`cad` dataset include subjects who have CAD result + `angio` dataset.

`whole` dataset include all the possible subjects.

### Finetune
```
python3 main.py --exp_name OURS_finetune_whole --dataset whole --trainset whole --testset cad --model CNN_B --batch_size 128 --phase finetune --ckpt_path OURS_pretrain --ckpt_epoch 5
```

### Linear Probing
```
python3 main.py --exp_name OURS_finetune_whole --dataset whole --trainset whole --testset cad --model CNN_B --batch_size 128 --phase finetune --ckpt_path OURS_pretrain --ckpt_epoch 5
```
