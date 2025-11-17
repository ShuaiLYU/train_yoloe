

\



# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# Activate ultralytics environment
conda activate clipenv





# for model in 26s; do


model=26s
lr=0.002
epo=30
close_mosaic=2
batch_size=128
clip_weight_name="mobileclip2:b" # mobileclip2b
timestamp=$(date +%Y%m%d_%H%M%S)

run_dir="runs"

ptw="object365v1" 

project_name=yoloe26s_tp
project_dir=${run_dir}/${project_name}
mkdir -p $project_dir
exp_name=${clip_weight_name}_${model}_bs${batch_size}_ptw${ptw}_cls${close_mosaic}_enginedata_gdobjv1_exp
exp_dir=${project_dir}/${exp_name}
echo "Experiment directory: $exp_dir"
mkdir -p $exp_dir
log_files="${exp_dir}-output.log"
timestamp=$(date +%Y%m%d_%H%M%S)
log_files="./runs/$timestamp.log"
echo "Log files: $log_files"

nohup python train_yoloe/finetune_yoloe26_tp_gdobjv1.py \
    --model_version $model \
    --lr $lr \
    --epochs $epo \
    --close_mosaic $close_mosaic \
    --batch $batch_size \
    --device 0,1,2,3 \
    --project $project_dir \
    --name $exp_name \
    --clip_weight_name $clip_weight_name \
    > $log_files 2>&1 &

echo "using the following command to check the log:\n tail -f -n 50 $log_files"




# todo. batch   close amp 
# --- IGNORE ---