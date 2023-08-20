for bs in 2048 4096; do
    for stage in 1 2 3 4 all '#1' '#2' '#3' resting SITTING; do
            for ckpt_epoch in {2..2}; do
            exp_name="SimCLR_${bs}_${ckpt_epoch}_1e-4_${stage}"
            cmd="CUDA_VISIBLE_DEVICES=1 python finetune.py --stage '${stage}' --exp_name ${exp_name} --datapath ./dataset/Severance --lr 0.0001 --test_batch ${bs} --ckpt_epoch ${ckpt_epoch} --lr 0.0001 > ./log/${exp_name}.out"
            echo "Running command: ${cmd}"
            eval ${cmd}
        done
    done
done