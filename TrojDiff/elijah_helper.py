#!/usr/bin/env python3
# coding=utf-8
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os

import torch
import numpy as np
import torch.utils.tensorboard as tb
from torchvision.utils import save_image
from torch import optim

from einops import rearrange, repeat
from piq import LPIPS

from runners.diffusion_attack_d2i import Diffusion

torch.set_printoptions(sci_mode=False)

TO_COMPUTE_TVLOSS = None
ALL_RES_DICT_FILENAME = None
ALL_RES_DICT = None


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--dataset", type=str, required=True
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default='ddpm',
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='./images/hello_kitty.png')
    parser.add_argument('--target_path', type=str, default='./images/mickey.png')
    parser.add_argument('--total_n_samples', type=int, default=50000)
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--remove_backdoor', action='store_true', help='remove backdoor')
    parser.add_argument('--merge_backdoor', action='store_true', help='remove backdoor')
    parser.add_argument("--ckpt_id", type=int, default=-1, help="ckpt_id to override the config.sampling.ckpt_id")
    parser.add_argument(
        "--do_not_contain_miu_target_path",
        action="store_true",
        help="do not contain miu path and target path in the log path for the old executions",
    )
    parser.add_argument('--compute_tvloss', action='store_true', help='compute tv loss instead of uniformity')

    args = parser.parse_args()

    global TO_COMPUTE_TVLOSS
    global ALL_RES_DICT_FILENAME
    global ALL_RES_DICT

    TO_COMPUTE_TVLOSS = args.compute_tvloss
    if TO_COMPUTE_TVLOSS:
        ALL_RES_DICT_FILENAME = './all_res_dict_tvloss.pt'
    else:
        ALL_RES_DICT_FILENAME = './all_res_dict.pt'
    if os.path.isfile(ALL_RES_DICT_FILENAME):
        ALL_RES_DICT = torch.load(ALL_RES_DICT_FILENAME)
    else:
        ALL_RES_DICT = dict()  # {generated_img_dir: loss, }

    if not args.exp:
        args.exp = os.path.join('./ddpm_attack_d2i',
                                'ft_cond_prob_' + str(args.cond_prob) + '_gamma_' + str(
                                    args.gamma) + '_target_label_' + str(args.target_label) + '_trigger_type_' + str(
                                    args.trigger_type))  # attack
        if args.trigger_type == 'patch':
            args.exp = args.exp + '_size_' + str(args.patch_size)

        if not args.do_not_contain_miu_target_path:
            # add miu and target info into the path
            args.exp = args.exp + '_' + os.path.basename(args.miu_path) + '_' + os.path.basename(args.target_path)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    if not args.test and not args.sample:
        # if not args.resume_training:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                shutil.rmtree(tb_path)
                os.makedirs(args.log_path)
                if os.path.exists(tb_path):
                    shutil.rmtree(tb_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    # NOTE: override the config.sampling.ckpt_id
    if args.ckpt_id != -1:
        new_config.sampling.ckpt_id = args.ckpt_id

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def compute_uniformity(images):
    # images = rearrange(images, 'b h w c -> b c h w')
    images1 = repeat(images, 'b c h w -> (b tile) c h w', tile=len(images))
    images2 = repeat(images, 'b c h w -> (tile b) c h w', tile=len(images))

    percept = LPIPS(replace_pooling=True, reduction="none")
    # NOTE: check if we need to resize the images to a certain size for LPIPS
    loss = percept(images1, images2).view(len(images), len(images))  # .mean()
    loss = torch.sort(loss, dim=1)[0]
    skip_cnt = 4
    loss = loss[:, skip_cnt:-skip_cnt]
    loss = loss.mean(dim=1)
    loss = torch.sort(loss)[0]
    loss = loss[skip_cnt:-skip_cnt].mean()

    # TODO: add MSE?

    return loss.item()


@torch.no_grad()
def sample_with_trigger(runner, trigger=None, use_trigger_input_options=(False, True), use_backdoor_sampling_options=(False, True), R_coef_T=None, recomp=False, save_res_dict=False):
    generated_img_dir = os.path.join(runner.args.log_path, f'generated_inverted_ckpt{runner.config.sampling.ckpt_id}{"_ema" if runner.config.model.ema else ""}_{R_coef_T}')

    if not os.path.isdir(generated_img_dir) or recomp:
        os.makedirs(generated_img_dir, exist_ok=True)

        before_fix = False
        before_merge = False
        model = runner.get_model(before_fix=before_fix, before_merge=before_merge)
        config = runner.config
        n = 16
        x0 = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=runner.device,
        )

        for use_trigger_input in use_trigger_input_options:
            for use_backdoor_sampling in use_backdoor_sampling_options:
                img_filename = os.path.join(generated_img_dir, 'sample')
                if use_trigger_input:
                    if trigger is not None:
                        x = x0 + trigger.to(x0.device)
                    else:
                        x = runner.args.gamma * x0 + runner.miu.unsqueeze(0).to(x0.device)  # N(miu,I)
                    img_filename = f'{img_filename}_trigger_input'
                else:
                    x = x0
                    img_filename = f'{img_filename}_benign_input'

                seq = range(0, runner.num_timesteps)
                from functions.denoising import ddpm_steps_bd, ddpm_steps
                if use_backdoor_sampling:
                    img_filename = f'{img_filename}_backdoor_sampling'
                    if not os.path.isfile(img_filename):
                        x = ddpm_steps_bd(x, seq, model, runner.betas, runner.miu, runner.coef_miu, runner.args)
                    else:
                        continue
                else:
                    img_filename = f'{img_filename}_benign_sampling'
                    if not os.path.isfile(img_filename):
                        x = ddpm_steps(x, seq, model, runner.betas)
                    else:
                        continue

                x = x[0][-1]

                from datasets import inverse_data_transform
                images = inverse_data_transform(config, x)

                # save_image(x, './tmp/backdoor_res_x+inverted_t.png')
                # save_image(x, './tmp/backdoor_res_benign_input.png')
                # save_image(x, './tmp/backdoor_model_backdoor_res_backdoor_input.png')
                # save_image(x, f'./tmp/{img_filename}.png')
                save_image(images, f'{img_filename}.png')
                torch.save(images.cpu(), f'{img_filename}.pt')

    if generated_img_dir not in ALL_RES_DICT or recomp:
        generated_img_ptfile = os.path.join(generated_img_dir, 'sample_trigger_input_backdoor_sampling.pt')
        images = torch.load(generated_img_ptfile)
        if TO_COMPUTE_TVLOSS:
            loss = compute_tvloss(images.cuda())
        else:
            loss = compute_uniformity(images.cuda())
        ALL_RES_DICT[generated_img_dir] = loss
    else:
        loss = ALL_RES_DICT[generated_img_dir]

    if save_res_dict:
        torch.save(ALL_RES_DICT, ALL_RES_DICT_FILENAME)

    print(f'{runner.args.log_path}{"_ema" if runner.config.model.ema else ""}@{R_coef_T}: {loss}')


from torchmetrics.image import TotalVariation
def compute_tvloss(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    tv = TotalVariation(reduction='mean').cuda()

    return tv(images).item()


def trigger_loss(noise, output):
    noise = noise.mean(0)
    output = output.mean(0)
    # save_image(noise.unsqueeze(0)*0.5+0.5, './tmp_noise.png')
    # save_image(output.unsqueeze(0)*0.5+0.5, './tmp_output.png')
    # print(noise.shape, output.shape)
    loss = torch.nn.functional.l1_loss(noise, output)
    return loss


def invert_trigger(runner):
    R_coef_T = 0.5
    trigger_filename = f'{runner.args.log_path}/trigger_ckpt{runner.config.sampling.ckpt_id}{"_ema" if runner.config.model.ema else ""}_{R_coef_T}.pt'

    if not os.path.isfile(trigger_filename):

        if not os.path.isdir(os.path.dirname(trigger_filename)):
            os.mkdir(os.path.dirname(trigger_filename))

        before_fix = False
        before_merge = False
        model = runner.get_model(before_fix=before_fix, before_merge=before_merge)
        config = runner.config
        n = 100
        noise = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=runner.device,
        )

        T = (torch.ones(n) * (runner.num_timesteps-1)).to(noise.device)

        def gen_mean_func(xT):
            return model(xT, T)

        # seq = range(0, runner.num_timesteps)
        # from functions.denoising import my_ddpm_one_step
        # gen_mean_func = my_ddpm_one_step(noise, seq, model, runner.betas, runner.miu, runner.coef_miu, runner.args)

        # with torch.no_grad():
        #     GT_trigger = runner.miu.to(runner.device).unsqueeze(0)
        #     trigger_noise = noise + GT_trigger
        #     model_output = gen_mean_func(trigger_noise)
        #     save_image(GT_trigger*0.5+0.5, './tmp_trigger_GT.png')
        #     save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output_GT.png')

        # R_coef_T = torch.tensor(1.).cuda()
        # R_coef_T = torch.tensor(0.5).cuda()

        trigger = torch.rand([1, 3, 32, 32]).cuda()
        trigger.requires_grad_(True)

        optimizer = optim.Adam([trigger, ], lr=0.1)

        # with torch.no_grad():
        #     trigger_noise = noise + trigger
        #     model_output = gen_mean_func(trigger_noise)
        #     save_image(trigger*0.5+0.5, './tmp_trigger_0.png')
        #     save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output_0.png')

        num_epochs = 100

        for epoch in range(num_epochs):

            optimizer.zero_grad()
            model.zero_grad()

            trigger_noise = noise + trigger
            model_output = gen_mean_func(trigger_noise)

            loss = trigger_loss(trigger*R_coef_T, model_output)

            loss.backward()
            optimizer.step()
            print(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')

        if not isinstance(R_coef_T, float):
            R_coef_T = R_coef_T.item()
        with torch.no_grad():
            # trigger_noise = noise + trigger
            # model_output = gen_mean_func(trigger_noise)
            # save_image(trigger*0.5+0.5, './tmp_trigger.png')
            # save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output.png')

            # trigger_filename = f'./trigger_{R_coef_T}.pt'
            torch.save(trigger.cpu(), trigger_filename)
    else:
        trigger = torch.load(trigger_filename, map_location='cpu')

    sample_with_trigger(runner, trigger, use_trigger_input_options=(True, ), use_backdoor_sampling_options=(True, ), R_coef_T=R_coef_T, save_res_dict=True)


def main():
    args, config = parse_args_and_config()
    logging.info("Doing trigger inversion")
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    assert args.sample
    assert args.sample_type == "ddpm_noisy" and args.skip_type == "uniform"

    try:
        runner = Diffusion(args, config)
        invert_trigger(runner)
        # trigger = None
        # trigger = torch.load('./trigger_0.030300000682473183.pt', map_location=runner.device)
        # trigger = torch.load('./trigger_1.0.pt', map_location=runner.device)
        # trigger = torch.load('./trigger_0.5.pt', map_location=runner.device)
        # trigger = 0
        # sample_with_trigger(runner, trigger)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
