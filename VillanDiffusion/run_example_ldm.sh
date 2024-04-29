#!/bin/bash

# invert the trigger and compute the uniformity score and tv loss.
python elijah_helper_ldm.py res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_epr0.0_GLASSES-CAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set_tmp --compute_tvloss
python elijah_helper_ldm.py res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_epr0.0_GLASSES-CAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set_tmp

# Use the inverted trigger to remove the injected backdoor.
python VillanDiffusion_rm.py --postfix new-set --project default --mode train --dataset CELEBA-HQ-LATENT --dataset_load_mode NONE --sde_type SDE-LDM --learning_rate 0.0002 --sched UNIPC-SCHED --infer_steps 20 --batch 8 --epoch 20 --clean_rate 0.1 --poison_rate 0. --trigger GLASSES --target CAT --solver_type ode --psi 1 --vp_scale 1.0 --ve_scale 1.0 --ckpt res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_epr0.0_GLASSES-CAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set_tmp --fclip o --save_image_epochs 1 --save_model_epochs 1 -o --gpu 0 --is_save_all_model_epochs --remove_backdoor
