#!/bin/bash

# invert the trigger and compute the uniformity score and tv loss.
python elijah_helper_ncsn.py res_NCSN_CIFAR10_my_CIFAR10_ep100_sde_c1.0_p0.5_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_flex_new-set --compute_tvloss
python elijah_helper_ncsn.py res_NCSN_CIFAR10_my_CIFAR10_ep100_sde_c1.0_p0.5_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_flex_new-set

# Use the inverted trigger to remove the injected backdoor.
python VillanDiffusion_rm.py --postfix flex_new-set --project default --mode train --learning_rate 2e-05 --dataset CIFAR10 --sde_type SDE-VE --batch 128 --epoch 11 --clean_rate 0.1 --poison_rate 0.0 --trigger STOP_SIGN_14 --target HAT --solver_type sde --psi 0 --vp_scale 1.0 --ve_scale 1.0 --ckpt res_NCSN_CIFAR10_my_CIFAR10_ep100_sde_c1.0_p0.5_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_flex_new-set --fclip o --save_image_epochs 1 --save_model_epochs 1 -o --dataset_load_mode FLEX --is_save_all_model_epochs --gpu 0 --remove_backdoor
