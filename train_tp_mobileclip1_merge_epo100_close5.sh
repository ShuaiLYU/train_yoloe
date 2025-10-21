




# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# Activate ultralytics environment
conda activate clipenv


clip_weight_name="mobileclip:blt" # mobileclip2b


for model in 11s
do
    lr=0.002

    timestamp=$(date +%Y%m%d_%H%M%S)

    run_dir="runs"
    run_dir=$(realpath $run_dir)

    epoch=1
    project_name=yoloe_original_train_tp
    project_dir=${run_dir}/${project_name}
    mkdir -p $project_dir
    exp_name=${clip_weight_name}_${model}_${lr}_close5_ep100__mergedata_exp
    exp_dir=${project_dir}/${exp_name}
    echo "Experiment directory: $exp_dir"
    mkdir -p $exp_dir
    log_files="${exp_dir}-output.log"
    echo "Log files: $log_files"


    nohup python train_yoloe/train_tp.py \
        --model_version $model \
        --lr $lr \
        --epochs 100 \
        --close_mosaic 5 \
        --batch 128 \
        --device 0,1,2,3 \
        --project $project_dir \
        --name $exp_name \
        --clip_weight_name $clip_weight_name \
        > $log_files 2>&1 &

        echo "using the following command to check the log:\n tail -f -n 50 $log_files"
done




# todo. batch   close amp 
# --- IGNORE ---