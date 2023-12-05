# python main.py --stage 1   --exp_name baseline_1   --lr 0.0001 --lr 0.0001 > ./log/baseline_1.out
# python main.py --stage 2   --exp_name baseline_2   --lr 0.0001 --lr 0.0001 > ./log/baseline_2.out
# python main.py --stage 3   --exp_name baseline_3   --lr 0.0001 --lr 0.0001 > ./log/baseline_3.out
# python main.py --stage 4   --exp_name baseline_4   --lr 0.0001 --lr 0.0001 > ./log/baseline_4.out
# python main.py --stage all --exp_name baseline_all --lr 0.0001 --lr 0.0001 > ./log/baseline_all.out

# for stage in '#1' '#2' '#3'; do
#     exp_name="baseline_${stage}"
#     cmd="python main.py --stage '${stage}' --exp_name ${exp_name} --lr 0.0001 > ./log/${exp_name}.out"
#     echo "Running command: ${cmd}"
#     eval ${cmd}
# done
conda activate aws_neuron_pytorch_p36
for stage in SITTING resting '#1' '#2' '#3'; do # 1 2 3 4 all SITTING resting '#1' '#2' '#3'; do
    exp_name="baseline_whole_${stage}"
    cmd="python main.py --stage '${stage}' --exp_name ${exp_name} --lr 0.0001 --batch_size 1024 --use_tb True --is_whole True > ./log/${exp_name}.out"
    echo "Running command: ${cmd}"
    eval ${cmd}
done