MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True " 
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --rescale_learned_sigmas True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8"
python scripts/image_train.py --train_mode train  --save_dir image_train65_20240918_0930 --use_P2_weight False --cubic_sampling False --data_source_dir /hdd/zhengyang/shiyan/guided-diffusion/imgs/ffhq/align256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

