#!/bin/bash

# invert the trigger and compute the uniformity score and tv loss.
python elijah_helper_ddim.py res_DDPM-CIFAR10-32_CIFAR10_ep10_ode_c1.0_p0.9_epr0.0_STOP_SIGN_14-HAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set --compute_tvloss
python elijah_helper_ddim.py res_DDPM-CIFAR10-32_CIFAR10_ep10_ode_c1.0_p0.9_epr0.0_STOP_SIGN_14-HAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set

# Use the inverted trigger to remove the injected backdoor.
python VillanDiffusion_rm.py --postfix new-set --project default --mode train --dataset CIFAR10 --sde_type SDE-VP --sched DDIM-SCHED --infer_steps 50 --batch 128 --epoch 50 --clean_rate 0.1 --poison_rate 0. --solver_type ode --psi 1 --vp_scale 1.0 --ve_scale 1.0 --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep10_ode_c1.0_p0.9_epr0.0_STOP_SIGN_14-HAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set --fclip o --save_image_epochs 1 --save_model_epochs 1 -o --gpu 0 --trigger STOP_SIGN_14 --target HAT  --is_save_all_model_epochs --dataset_load_mode FLEX  --remove_backdoor --learning_rate 2e-5
