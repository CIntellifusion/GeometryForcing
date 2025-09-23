mkdir checkpoints
hf download Haoyuwu/GeometryForcing --include "DFoT_16f_state_dict.ckpt" --local-dir ./checkpoints
hf download Haoyuwu/GeometryForcing --include "geometry_forcing_state_dict.ckpt" --local-dir ./checkpoints
hf download Haoyuwu/GeometryForcing --include "geometry_forcing_with_dino_state_dict.ckpt" --local-dir ./checkpoints