python finetune.py --stage 1   --exp_name BYOL_2048_3_1   --lr 0.0001 --test_batch 2048 --ckpt_epoch 3 > ./log/BYOL_2048_3_1.out
python finetune.py --stage 2   --exp_name BYOL_2048_3_2   --lr 0.0001 --test_batch 2048 --ckpt_epoch 3 > ./log/BYOL_2048_3_2.out
python finetune.py --stage 3   --exp_name BYOL_2048_3_3   --lr 0.0001 --test_batch 2048 --ckpt_epoch 3 > ./log/BYOL_2048_3_3.out
python finetune.py --stage 4   --exp_name BYOL_2048_3_4   --lr 0.0001 --test_batch 2048 --ckpt_epoch 3 > ./log/BYOL_2048_3_4.out
python finetune.py --stage all --exp_name BYOL_2048_3_all --lr 0.0001 --test_batch 2048 --ckpt_epoch 3 > ./log/BYOL_2048_3_all.out

python finetune.py --stage 1   --exp_name BYOL_4096_3_1   --lr 0.0001 --test_batch 4096 --ckpt_epoch 3 > ./log/BYOL_4096_3_1.out
python finetune.py --stage 2   --exp_name BYOL_4096_3_2   --lr 0.0001 --test_batch 4096 --ckpt_epoch 3 > ./log/BYOL_4096_3_2.out
python finetune.py --stage 3   --exp_name BYOL_4096_3_3   --lr 0.0001 --test_batch 4096 --ckpt_epoch 3 > ./log/BYOL_4096_3_3.out
python finetune.py --stage 4   --exp_name BYOL_4096_3_4   --lr 0.0001 --test_batch 4096 --ckpt_epoch 3 > ./log/BYOL_4096_3_4.out
python finetune.py --stage all --exp_name BYOL_4096_3_all --lr 0.0001 --test_batch 4096 --ckpt_epoch 3 > ./log/BYOL_4096_3_all.out