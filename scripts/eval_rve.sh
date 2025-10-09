echo "Running evaluation for RVE"
cd evaluation

revisit_eval_out_base="outputs_rve"
all_setting_names=(
    "GeometryForcingREPA"
)
all_gif_dirs=(
    "output/evaluations/{put_your_run_name_here}/wandb/latest-run/files/media/videos/prediction_vis"
)

for i in "${!all_setting_names[@]}"; do
    setting_name=${all_setting_names[i]}
    gif_dir=${all_gif_dirs[i]}
    
    # Create log directory if it doesn't exist
    log_dir=${revisit_eval_out_base}/${setting_name}
    mkdir -p ${log_dir}

    temp_dir=${log_dir}/"video_vis"
    output_dir=${log_dir}/"eval"
    

    python eval_revisiting.py \
        --temp_dir ${temp_dir} \
        --output_dir ${output_dir} \
        --gif_dir ${gif_dir} \
        > ${log_dir}/log.txt 2>&1
done