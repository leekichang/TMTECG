# python finetune.py --stage 1   --exp_name BYOL_2048_1_1e-4_1   --lr 0.0001 --test_batch 2048 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_2048_1_1e-4_1.out
# python finetune.py --stage 2   --exp_name BYOL_2048_1_1e-4_2   --lr 0.0001 --test_batch 2048 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_2048_1_1e-4_2.out
# python finetune.py --stage 3   --exp_name BYOL_2048_1_1e-4_3   --lr 0.0001 --test_batch 2048 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_2048_1_1e-4_3.out
# python finetune.py --stage 4   --exp_name BYOL_2048_1_1e-4_4   --lr 0.0001 --test_batch 2048 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_2048_1_1e-4_4.out
# python finetune.py --stage all --exp_name BYOL_2048_1_1e-4_all --lr 0.0001 --test_batch 2048 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_2048_1_1e-4_all.out

# python finetune.py --stage 1   --exp_name BYOL_4096_1_1e-4_1   --lr 0.0001 --test_batch 4096 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_4096_1_1e-4_1.out
# python finetune.py --stage 2   --exp_name BYOL_4096_1_1e-4_2   --lr 0.0001 --test_batch 4096 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_4096_1_1e-4_2.out
# python finetune.py --stage 3   --exp_name BYOL_4096_1_1e-4_3   --lr 0.0001 --test_batch 4096 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_4096_1_1e-4_3.out
# python finetune.py --stage 4   --exp_name BYOL_4096_1_1e-4_4   --lr 0.0001 --test_batch 4096 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_4096_1_1e-4_4.out
# python finetune.py --stage all --exp_name BYOL_4096_1_1e-4_all --lr 0.0001 --test_batch 4096 --ckpt_epoch 1 --lr 0.0001 > ./log/BYOL_4096_1_1e-4_all.out

# python finetune.py --stage 1   --exp_name BYOL_2048_2_1e-4_1   --lr 0.0001 --test_batch 2048 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_2048_2_1e-4_1.out
# python finetune.py --stage 2   --exp_name BYOL_2048_2_1e-4_2   --lr 0.0001 --test_batch 2048 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_2048_2_1e-4_2.out
# python finetune.py --stage 3   --exp_name BYOL_2048_2_1e-4_3   --lr 0.0001 --test_batch 2048 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_2048_2_1e-4_3.out
# python finetune.py --stage 4   --exp_name BYOL_2048_2_1e-4_4   --lr 0.0001 --test_batch 2048 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_2048_2_1e-4_4.out
# python finetune.py --stage all --exp_name BYOL_2048_2_1e-4_all --lr 0.0001 --test_batch 2048 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_2048_2_1e-4_all.out

# conda activate aws_neuron_pytorch_p36
# for bs in 2048 4096; do
#     for stage in 1 2 3 4 all '#1' '#2' '#3' resting SITTING; do
#             for ckpt_epoch in {96..96}; do
#             exp_name="BYOL_${bs}_whole_${ckpt_epoch}_1e-4_${stage}"
#             cmd="python finetune.py --stage '${stage}' --exp_name ${exp_name} --lr 0.0001 --batch_size 1024 --test_batch ${bs} --ckpt_epoch ${ckpt_epoch} --lr 0.0001 --use_tb True --is_whole True > ./log/${exp_name}.out"
#             echo "Running command: ${cmd}"
#             eval ${cmd}
#         done
#     done
# done

conda activate aws_neuron_pytorch_p36
for bs in 2048 4096; do
    for stage in '#1' '#2' '#3' resting SITTING; do #1 2 3 4 all 
            for ckpt_epoch in {96..96}; do
            exp_name="BYOL_${bs}_${ckpt_epoch}_1e-4_${stage}"
            cmd="python finetune.py --stage '${stage}' --exp_name ${exp_name} --lr 0.0001 --batch_size 128 --test_batch ${bs} --ckpt_epoch ${ckpt_epoch} --lr 0.0001 --use_tb True --is_whole False > ./log/${exp_name}.out"
            echo "Running command: ${cmd}"
            eval ${cmd}
        done
    done
done

# python finetune.py --stage 1   --exp_name BYOL_4096_2_1e-4_1   --lr 0.0001 --test_batch 4096 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_4096_2_1e-4_1.out
# python finetune.py --stage 2   --exp_name BYOL_4096_2_1e-4_2   --lr 0.0001 --test_batch 4096 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_4096_2_1e-4_2.out
# python finetune.py --stage 3   --exp_name BYOL_4096_2_1e-4_3   --lr 0.0001 --test_batch 4096 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_4096_2_1e-4_3.out
# python finetune.py --stage 4   --exp_name BYOL_4096_2_1e-4_4   --lr 0.0001 --test_batch 4096 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_4096_2_1e-4_4.out
# python finetune.py --stage all --exp_name BYOL_4096_2_1e-4_all --lr 0.0001 --test_batch 4096 --ckpt_epoch 2 --lr 0.0001 > ./log/BYOL_4096_2_1e-4_all.out