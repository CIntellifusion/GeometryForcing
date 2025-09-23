mkdir checkpoints
modelscope download --model HaoyuWuRUC/GeometryForcing "DFoT_16f_state_dict.ckpt"  --local_dir ./checkpoints
modelscope download --model HaoyuWuRUC/GeometryForcing "geometry_forcing_state_dict.ckpt"  --local_dir ./checkpoints
modelscope download --model HaoyuWuRUC/GeometryForcing "geometry_forcing_with_dino_state_dict.ckpt"  --local_dir ./checkpoints