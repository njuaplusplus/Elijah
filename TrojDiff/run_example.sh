#!/bin/bash
target_path="mickey"
miu_path="white"

export CUDA_VISIBLE_DEVICES=0

# backdoor model
python main_attack_d2i.py --dataset cifar10 --config cifar10_100k.yml --target_label 7 --ni --resume_training --target_label 7 --gamma 0.1 --trigger_type patch --miu_path "./images/${miu_path}.png" --target_path "./images/${target_path}.png" --patch_size 3 --max_steps 100000

# invert trigger
python elijah_helper.py --dataset cifar10 --config cifar10_no_ema.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path "./images/${miu_path}.png" --target_path "./images/${target_path}.png" --patch_size 3

# backdoor removal
python main_attack_d2i.py --dataset cifar10 --config cifar10_no_ema.yml --target_label 7 --ni --resume_training --target_label 7 --gamma 0.1 --trigger_type patch --miu_path "./images/${miu_path}.png" --target_path "./images/${target_path}.png" --patch_size 3 --remove_backdoor --subset_ratio 0.1  --max_steps=20000

# sample with backdoor-removed model 
ckpt_id=10000
python main_attack_d2i.py --dataset cifar10 --config cifar10_no_ema.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path "./images/${miu_path}.png" --target_path "./images/${target_path}.png" --patch_size 3 --exp "./ddpm_attack_d2i/ft_cond_prob_1.0_gamma_0.1_target_label_7_trigger_type_patch_size_3_${miu_path}.png_${target_path}.png_remove_backdoor" --ckpt_id $ckpt_id

# sample with backdoored model
ckpt_id=100000
python main_attack_d2i.py --dataset cifar10 --config cifar10_no_ema.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path "./images/${miu_path}.png" --target_path "./images/${target_path}.png" --patch_size 3 --ckpt_id $ckpt_id
