#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from einops import rearrange, repeat
from piq import LPIPS

import torch
from torch import optim
from torchvision.utils import save_image
from torchmetrics.image import TotalVariation

from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler
from model_rm import DiffuserModelSched
from util import Samples


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def format_ckpt_dir(ckpt):
    return ckpt.replace('/', '_')


@torch.no_grad()
def sample_with_trigger(ckpt, trigger, file_name, R_coef_T, recomp=False, use_ddim=False, save_res_dict=False, to_encode_trigger=False):
    assert not use_ddim
    clip = False
    clip_opt = "_noclip"
    test_dir = os.path.join('./generated_img_with_trigger/', format_ckpt_dir(ckpt))
    os.makedirs(test_dir, exist_ok=True)
    generated_img_ptfile = os.path.join(test_dir, f'{file_name}{clip_opt}.pt')

    if not os.path.isfile(generated_img_ptfile) or recomp:
        if use_ddim:
            raise AssertionError('not used here!')
            model, noise_sched, get_pipeline = DiffuserModelSched.new_get_pretrained(ckpt=ckpt, clip_sample=clip, noise_sched_type=DiffuserModelSched.DDIM_SCHED)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = get_pipeline(unet=unet, scheduler=noise_sched)
        else:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ckpt, clip_sample=clip, noise_sched_type=DiffuserModelSched.UNIPC_SCHED, sde_type=DiffuserModelSched.SDE_LDM)
            unet = model.cuda()
            vae = vae.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = get_pipeline(accelerate=None, unet=unet, vae=vae, scheduler=noise_sched, no_accelerate=True)

        save_gif = False

        def gen_samples(init):

            # Sample some images from random noise (this is the backward diffusion process).
            # The default pipeline output type is `List[PIL.Image]`
            if use_ddim:
                pipline_res = pipeline(
                    batch_size = 16,
                    init=init,
                    output_type=None,
                    num_inference_steps=50
                )
            else:
                pipline_res = pipeline(
                    batch_size = 16,
                    init=init,
                    output_type=None,
                    num_inference_steps=20,
                    return_full_mov=save_gif
                )

            images = pipline_res.images
            movie = pipline_res.movie

            print(type(images), images.shape, images.max(), images.min())
            print(type(np.array(movie)), np.array(movie).shape)
            # <class 'numpy.ndarray'> (16, 32, 32, 3) 1.0 0.037254035
            if TO_COMPUTE_TVLOSS:
                loss = compute_tvloss(torch.from_numpy(images).cuda())
            else:
                loss = compute_uniformity(torch.from_numpy(images).cuda())
            ALL_RES_DICT[ckpt][R_coef_T] = loss
            torch.save(torch.from_numpy(images), generated_img_ptfile)

            # images_np = images

            # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]

            # # Make a grid out of the images
            image_grid = make_grid(images, rows=4, cols=4)

            if save_gif:
                sam_obj = Samples(samples=np.array(movie), save_dir=test_dir)

            # # Save the images
            image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
            if save_gif:
                sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
                sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)

        with torch.no_grad():
            print('pipeline.device', pipeline.device)
            print('pipeline.vqvae.device', pipeline.vqvae.device)
            noise = torch.randn([16, ] + noise_shape)
            print('noise.shape', noise.shape)
            # NOTE: here trigger is after encoding
            if to_encode_trigger:
                assert hasattr(pipeline, 'encode')
                trigger = pipeline.encode(trigger.to(pipeline.device)).to(noise.device)
                print('trigger.shape', trigger.shape)
            init = noise + trigger
            # init = trigger
            # init = pipeline.scheduler.sigmas[0] * (noise + trigger)
            gen_samples(init=init)

            # villan_gen_samples(init=init, folder=test_dir, pipeline=pipeline)
    elif R_coef_T not in ALL_RES_DICT[ckpt]:
        images = torch.load(generated_img_ptfile)
        if TO_COMPUTE_TVLOSS:
            loss = compute_tvloss(images.cuda())
        else:
            loss = compute_uniformity(images.cuda())
        ALL_RES_DICT[ckpt][R_coef_T] = loss
    else:
        pass

    print(f'{ckpt}@{R_coef_T}: {ALL_RES_DICT[ckpt][R_coef_T]}')

    # NOTE: skip save so that we can parallel run them
    if save_res_dict:
        torch.save(ALL_RES_DICT, ALL_RES_DICT_FILENAME)


def trigger_loss(noise, output):
    noise = noise.mean(0)
    output = output.mean(0)
    # save_image(noise.unsqueeze(0)*0.5+0.5, './tmp_noise.png')
    # save_image(output.unsqueeze(0)*0.5+0.5, './tmp_output.png')
    # print(noise.shape, output.shape)
    loss = torch.nn.functional.l1_loss(noise, output)
    return loss


def opt_r(ckpt):
    clip = False

    R_coef_T = 0.5
    trigger_filename = f'./inverted_trigger/{format_ckpt_dir(ckpt)}_trigger_{R_coef_T}.pt'

    if not os.path.isdir(os.path.dirname(trigger_filename)):
        os.mkdir(os.path.dirname(trigger_filename))

    if not os.path.isfile(trigger_filename):
    # if True:
        model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ckpt, clip_sample=clip, noise_sched_type=DiffuserModelSched.UNIPC_SCHED, sde_type=DiffuserModelSched.SDE_LDM)
        unet = model.cuda()
        vae = vae.cuda()
        noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
        pipeline = get_pipeline(accelerate=None, unet=unet, vae=vae, scheduler=noise_sched, no_accelerate=True)

        if noise_shape[-1] == 256:
            bs = 20
        elif noise_shape[-1] == 128:
            bs = 50
        else:
            bs = 100
        bs = 40
        noise = torch.randn([bs, ] + noise_shape).cuda()
        T = noise_sched.num_train_timesteps-1

        print('R_coef_T', R_coef_T)

        # trigger = -torch.rand([1, 3, 256, 256]).cuda()
        trigger = torch.rand([1, 3, 256, 256]).cuda() * 2 - 1
        print('trigger stat:', trigger.min(), trigger.max())
        trigger = pipeline.encode(trigger)
        del vae
        del pipeline
        trigger.requires_grad_(True)
        optimizer = optim.Adam([trigger, ], lr=0.1)

        num_epochs = 10

        for epoch in range(num_epochs):

            optimizer.zero_grad()
            unet.zero_grad()

            trigger_noise = noise + trigger
            model_output = unet(trigger_noise, T).sample

            #  / noise_sched.init_noise_sigma
            loss = trigger_loss(2*trigger*R_coef_T, model_output)

            loss.backward()
            optimizer.step()
            print(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')

        if not isinstance(R_coef_T, float):
            R_coef_T = R_coef_T.item()
        with torch.no_grad():
            # trigger_noise = noise + trigger
            # model_output = unet(trigger_noise, T).sample
            # save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output.png')

            # torch.save(trigger.cpu(), f'./{ckpt}_trigger.pt')
            torch.save(trigger.cpu(), trigger_filename)
    else:
        trigger = torch.load(trigger_filename, map_location='cpu')

    filename = f'{format_ckpt_dir(ckpt)}_inverted_{R_coef_T}'
    # sample_with_trigger(ckpt, trigger.cpu(), filename, R_coef_T, recomp=True, save_res_dict=False)
    sample_with_trigger(ckpt, trigger.cpu(), filename, R_coef_T, recomp=False, save_res_dict=True)


def compute_uniformity(images):
    images = rearrange(images, 'b h w c -> b c h w')
    images1 = repeat(images, 'b c h w -> (b tile) c h w', tile=len(images))
    images2 = repeat(images, 'b c h w -> (tile b) c h w', tile=len(images))

    percept = LPIPS(replace_pooling=True, reduction="none")
    loss = percept(images1, images2).view(len(images), len(images))
    loss = torch.sort(loss, dim=1)[0]
    skip_cnt = 4
    loss = loss[:, skip_cnt:-skip_cnt]
    loss = loss.mean(dim=1)
    loss = torch.sort(loss)[0]
    loss = loss[skip_cnt:-skip_cnt].mean()

    return loss.item()


def compute_tvloss(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    tv = TotalVariation(reduction='mean').cuda()

    return tv(images).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tvloss', action='store_true', help='compute tv loss instead of uniformity')
    parser.add_argument('ckpt', help='checkpoint')
    args = parser.parse_args()

    TO_COMPUTE_TVLOSS = args.compute_tvloss
    if TO_COMPUTE_TVLOSS:
        ALL_RES_DICT_FILENAME = './all_res_dict_tvloss.pt'
    else:
        ALL_RES_DICT_FILENAME = './all_res_dict.pt'
    if os.path.isfile(ALL_RES_DICT_FILENAME):
        ALL_RES_DICT = torch.load(ALL_RES_DICT_FILENAME)
    else:
        ALL_RES_DICT = defaultdict(dict) #{ckpt: {R_coef_T: loss, }, }

    opt_r(args.ckpt)
