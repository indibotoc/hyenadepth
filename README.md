The train command used (reconstructed):

/home/ubuntu/anaconda3/envs/hyenapixel/bin/python3 train.py
\
--model_name hpx_guided \
--encoder_type hyenapixel \
--use_guided_upsample \
--data_path ../kitti_data \
--log_dir ../logs \
--batch_size 6 \
--num_workers 12 \
--png \
--wandb_project hyenapixel-depth

- Eval command:

/home/ubuntu/anaconda3/envs/hyenapixel/bin/python3
evaluate_depth.py \
--load_weights_folder
../logs/hpx_guided/models/weights_19 \
--eval_mono \
--data_path ../kitti_data \
--encoder_type hyenapixel \
--use_guided_upsample \
--png \
--num_workers 4 \
--wandb_project hyenapixel-depth