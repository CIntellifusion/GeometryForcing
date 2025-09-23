
dataset="realestate10k_rotate" # "realestate10k"
algorithm="dfot_geometry_forcing"

exp_name="eval_360_rotation"
result_dir="$output_dir/train/$exp_name"
checkpoint_path="checkpoints/geometry_forcing_state_dict.ckpt"
eval_result_dir="$output_dir/geometry_forcing_rotation"

echo "Result directory: $result_dir"
echo "Checkpoint path: $checkpoint_path"
# ## construct evaluation args
checkpoints_dir=${result_dir}/checkpoints

python -m main +name=rotation dataset=$dataset \
        algorithm=$algorithm \
        experiment=video_generation @diffusion/continuous \
        algorithm.alignment.apply_unnormalize_recon=True \
        algorithm.alignment.latents_info=2 \
        load=$checkpoint_path \
        dataset.num_eval_videos=1000 \
        'experiment.tasks=[validation]' experiment.validation.data.shuffle=False experiment.test.data.shuffle=False dataset.frame_skip=1 \
        dataset.n_frames=16 \
        algorithm.logging.max_num_videos=1000 \
        dataset.context_length=1 \
        algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla \
        +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 \
        +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  \
        experiment.validation.batch_size=1 \
        algorithm.tasks.interpolation.history_guidance.name=vanilla \
        +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5 \
        'algorithm.logging.metrics=[fvd,fid,psnr,lpips,ssim]' 
        hydra.run.dir=$eval_result_dir 