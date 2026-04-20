MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000  --noise_schedule linear --learn_sigma True --rescale_learned_sigmas True" 
python scripts/image_sample.py --isSample True --model_path ./logs/image_train65/ema_0.9999.pt --scale 0 --output_path_dir image_sample65 --output_path_dir_type mask --output_pt 390000  $MODEL_FLAGS $DIFFUSION_FLAGS 
