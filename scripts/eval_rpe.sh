echo "Running evaluation for RPE"
cd evaluation

# base output directory
rpe_out_base="outputs_rpe"

# Define arrays for different settings
all_setting_names=(
    "GeometryForcingREPA"
)
all_gif_dirs=(
    "output/evaluations/{put_your_run_name_here}/wandb/latest-run/files/media/videos/prediction_vis"
)
all_gpu_ids=(
    "0"
)

for i in "${!all_setting_names[@]}"; do
    setting_name=${all_setting_names[i]}
    gif_dir=${all_gif_dirs[i]}
    gpu_id=${all_gpu_ids[i]}
    
    # Create log directory if it doesn't exist
    log_dir=${rpe_out_base}/${setting_name}
    mkdir -p ${log_dir}
    temp_dir=${log_dir}/"video_vis"
    output_dir=${log_dir}/"eval"

    echo "Running ${setting_name} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python eval_rpe.py \
        --temp_dir ${temp_dir} \
        --output_dir ${output_dir} \
        --gif_dir ${gif_dir} \
        > ${log_dir}/log.txt 2>&1
done

# Wait for all background processes to complete
wait