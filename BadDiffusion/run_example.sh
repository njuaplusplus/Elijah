#!/bin/bash

# Train a backdoored model. By default, it's saved as res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-HAT
python baddiffusion.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger STOP_SIGN_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0

# invert the trigger and compute the uniformity score and tv loss
python elijah_helper.py res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-HAT --compute_tvloss
python elijah_helper.py res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-HAT

# Use the inverted trigger to remove the injected backdoor. By default, it's aved as res_res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-HAT_CIFAR10_ep11_c0.1_p0.0_STOP_SIGN_14-HAT
python baddiffusion.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 11 --poison_rate 0 --clean_rate 0.1 --trigger STOP_SIGN_14 --target HAT --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-HAT --fclip o -o --gpu 0 --save_image_epochs 1 --save_model_epochs 1 --is_save_all_model_epochs --dataset_load_mode FLEX --remove_backdoor
