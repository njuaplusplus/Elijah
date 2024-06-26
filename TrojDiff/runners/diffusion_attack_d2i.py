import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses_attack_d2dout import loss_registry
from functions.losses import loss_registry as benign_loss_registry
from datasets import get_dataset, get_targetset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import torchvision.transforms as T

from PIL import Image


def cycle(dl):
    while True:
        for data in dl:
            yield data


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # attack
        miu = Image.open(args.miu_path)
        transform = T.Compose([
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor()
        ])
        miu = transform(miu)  # [0,1]
        miu = data_transform(self.config, miu)  # [-1,1]
        miu = miu * (1 - args.gamma)  # [-0.5,0.5]
        self.miu = miu  # (3,32,32)

        k_t = torch.randn_like(betas)
        for ii in range(config.diffusion.num_diffusion_timesteps):
            tmp_sum = torch.sqrt(1. - alphas_cumprod[ii])
            tmp_alphas = torch.flip(alphas[:ii + 1], [0])
            for jj in range(1, ii + 1):
                tmp_sum -= k_t[ii - jj] * torch.sqrt(torch.prod(tmp_alphas[:jj]))
            k_t[ii] = tmp_sum
        coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(alphas) * k_t
        self.coef_miu = coef_miu

        target_img = Image.open(args.target_path)
        target_img = transform(target_img)  # [0,1]
        self.target_img = target_img  # (3,32,32)


    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # base dataset
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)

        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            # attack: resume from pre-trained model
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                states = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(states)
                model = torch.nn.DataParallel(model)
                if self.config.model.ema:
                    ema_helper.load_state_dict(states)
            elif self.config.data.dataset == "CELEBA":
                ckpt = '/home/username/ddim/saved/celeba_ckpt.pth'
                states = torch.load(ckpt, map_location=self.device)[4]
                model.load_state_dict(states)
                model = torch.nn.DataParallel(model)
                if self.config.model.ema:
                    ema_helper.load_state_dict(states)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # target data
                target_bs = int(x.shape[0]*0.1)
                x_tar = torch.stack([self.target_img] * target_bs)  # (batch,3,32,32)
                y_tar = torch.ones(target_bs) * 1000 # (batch)
                x = torch.cat([x, x_tar], dim=0)
                y= torch.cat([y, y_tar], dim=0)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                y = y.to(self.device)

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, y, t, e, b, self.miu, self.args)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)


                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                if step == self.args.max_steps:
                    return

                data_start = time.time()


    def merge_backdoor(self):
        self.remove_backdoor(trigger=None)

    def remove_backdoor(self, trigger=None):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if trigger is None:
            assert args.merge_backdoor, 'merge the backdoor sampling and benign sampling'
            trigger = 0

        # base dataset
        # NOTE: to reduece to 10% dataset
        dataset, test_dataset = get_dataset(args, config)
        logging.info(f'dataset length: {len(dataset)}')
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)

        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0

        model_ckpt_dir = self.args.log_path
        if self.args.remove_backdoor:
            model_ckpt_dir = model_ckpt_dir.replace('_remove_backdoor', '')

        if isinstance(trigger, int):
            # when trigger is 0, we merge the backdoor and benign from original backdoored model
            # otherwise, it means we are trying to remove backdoor from merge backdoor
            #            where we need to set both --remove_backdoor and --merge_backdoor
            model_ckpt_dir = model_ckpt_dir.replace('_merge_backdoor', '')

        # attack: resume from pre-trained model
        if getattr(self.config.sampling, "ckpt_id", None) is None:
            model_pt_file = os.path.join(model_ckpt_dir, "ckpt.pth")
            states = torch.load(
                model_pt_file,
                map_location=self.config.device,
            )
        else:
            model_pt_file = os.path.join(model_ckpt_dir, f"ckpt_{self.config.sampling.ckpt_id}.pth")
            states = torch.load(
                model_pt_file,
                map_location=self.config.device,
            )
        model = model.to(self.device)
        logging.info(f'Loading model from {model_pt_file}')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            logging.info('using ema')
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        before_merge = not args.remove_backdoor

        @torch.no_grad()
        def get_frozen_model(before_fix=True, before_merge=before_merge):
            model = self.get_model(before_fix=before_fix, before_merge=before_merge)
            for p in model.parameters():
                p.requires_grad_(False)
            return model

        frozen_model = get_frozen_model()

        seq = range(0, self.num_timesteps)
        from functions.denoising import my_ddpm_one_step, my_benign_ddpm_one_step
        # define deshift loss

        def deshift_loss(noise, gen_mean_func_backdoor, gen_mean_func_benign, gen_mean_func_benign_frozen):
            # goal is to make sure the output of gen_mean_func_backdoor and gen_mean_func_benign is similar
            benign_prediction = gen_mean_func_benign(noise)
            with torch.no_grad():
                frozen_benign_prediction = gen_mean_func_benign_frozen(noise)
            if isinstance(trigger, int):
                assert trigger == 0
                backdoor_prediction = gen_mean_func_backdoor(noise)
            else:
                backdoor_prediction = gen_mean_func_backdoor(noise+trigger.to(noise.device))
            loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
            loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
            return loss1, loss2

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                # y = y.to(self.device)

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # loss0 = 0.

                # loss0 = loss_registry[config.model.type](model, x, y, t, e, b, self.miu, self.args)

                loss0 = benign_loss_registry[config.model.type](model, x, t, e, b)

                noise = e
                if not self.config.model.ema or isinstance(trigger, int):
                    # if not ema, use this loss
                    # if merge backdoor, use this loss
                    gen_mean_func_backdoor = my_ddpm_one_step(noise, seq, model, self.betas, self.miu, self.coef_miu, self.args)
                    gen_mean_func_benign = my_benign_ddpm_one_step(noise, seq, model, self.betas, self.miu, self.coef_miu, self.args)
                    gen_mean_func_benign_frozen = my_benign_ddpm_one_step(noise, seq, frozen_model, self.betas, self.miu, self.coef_miu, self.args)
                else:
                    # remove backdoor
                    def gen_mean_func_benign(xT):
                        return model(xT, (torch.ones(n) * (self.num_timesteps-1)).to(noise.device))
                    gen_mean_func_backdoor = gen_mean_func_benign
                    def gen_mean_func_benign_frozen(xT):
                        return frozen_model(xT, (torch.ones(n) * (self.num_timesteps-1)).to(noise.device))

                loss1, loss2 = deshift_loss(e, gen_mean_func_backdoor, gen_mean_func_benign, gen_mean_func_benign_frozen)

                loss = loss0 + loss1 + loss2

                tb_logger.add_scalar("loss", loss, global_step=step)
                tb_logger.add_scalar("loss0", loss0, global_step=step)
                tb_logger.add_scalar("loss1", loss1, global_step=step)
                tb_logger.add_scalar("loss2", loss2, global_step=step)

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item()}, loss0: {loss0 and loss0.item()},"
                    f"loss1: {loss1 and loss1.item()}, loss2: {loss2 and loss2.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)


                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                if step == self.args.max_steps:
                    return

                data_start = time.time()


    @torch.no_grad()
    def sample(self):
        # model = Model(self.config)

        # if not self.args.use_pretrained:
        #     if getattr(self.config.sampling, "ckpt_id", None) is None:
        #         states = torch.load(
        #             os.path.join(self.args.log_path, "ckpt.pth"),
        #             map_location=self.config.device,
        #         )
        #     else:
        #         states = torch.load(
        #             os.path.join(
        #                 self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
        #             ),
        #             map_location=self.config.device,
        #         )
        #     model = model.to(self.device)
        #     print("Loading model...")
        #     model = torch.nn.DataParallel(model)
        #     model.load_state_dict(states[0], strict=True)

        #     if self.config.model.ema:
        #         ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        #         ema_helper.register(model)
        #         ema_helper.load_state_dict(states[-1])
        #         ema_helper.ema(model)
        #     else:
        #         ema_helper = None
        # else:
        #     # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        #     if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
        #         if self.config.data.dataset == "CIFAR10":
        #             name = "cifar10"
        #         elif self.config.data.dataset == "LSUN":
        #             name = f"lsun_{self.config.data.category}"
        #         else:
        #             raise ValueError
        #         ckpt = get_ckpt_path(f"ema_{name}")
        #         print("Loading checkpoint {}".format(ckpt))
        #         model.load_state_dict(torch.load(ckpt, map_location=self.device))
        #         model.to(self.device)
        #         model = torch.nn.DataParallel(model)

        #     elif self.config.data.dataset == "CELEBA":
        #         ckpt = '/home/username/ddim/saved/celeba_ckpt.pth'
        #         states = torch.load(ckpt, map_location=self.device)[4]
        #         model.load_state_dict(states)
        #         model.to(self.device)
        #         model = torch.nn.DataParallel(model)

        # model.eval()

        model = self.get_model(before_fix=False, before_merge=False)

        if self.args.fid:
            if self.args.bd_but_benign_input:
                self.sample_fid_bd(model, use_benign_input=True)
                self.sample_fid_bd(model)
            else:
                self.sample_fid_bd(model)
                self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence_bd(model)
            # self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config

        if self.args.use_pretrained:
            image_folder = self.args.image_folder + '_pretrained'
        else:
            image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id)
        if self.args.eta != 1: # ddim
            image_folder = image_folder + '_ddim_eta_' + str(self.args.eta)
        image_folder = image_folder + '_' + self.args.dataset
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = len(glob.glob(f"{image_folder}/*.png"))
        # img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = self.args.total_n_samples
        # total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        print(f'n_rounds: {n_rounds}')

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_fid_bd(self, model, use_benign_input=False):
        config = self.config
        if self.args.use_pretrained:
            image_folder = self.args.image_folder + '_pretrained_bd'
        else:
            image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id) + '_bd'
        if self.args.eta != 1: # ddim
            image_folder = image_folder + '_ddim_eta_' + str(self.args.eta)
        image_folder = image_folder + '_' + self.args.dataset
        if use_benign_input:
            image_folder = image_folder + '_use_benign_input'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = len(glob.glob(f"{image_folder}/*.png"))
        # img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = self.args.total_n_samples
        # total_n_samples = 1000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        print(f'n_rounds: {n_rounds}')

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                if not use_benign_input:
                    miu = torch.stack([self.miu.to(self.device)] * n)  # (batch,3,32,32)
                    tmp_x = x.clone()
                    x = self.args.gamma * x + miu  # N(miu,I)
                    if self.args.trigger_type == 'patch':
                        tmp_x[:, :, -self.args.patch_size:, -self.args.patch_size:] = x[:, :, -self.args.patch_size:, -self.args.patch_size:]
                        x = tmp_x

                x = self.sample_image_bd(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_sequence_bd(self, model):
        config = self.config

        n = 4
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        miu = torch.stack([self.miu.to(self.device)] * n)  # (batch,3,32,32)
        tmp_x = x.clone()
        x = self.args.gamma * x + miu  # N(miu,I)
        if self.args.trigger_type == 'patch':
            tmp_x[:, :, -self.args.patch_size:, -self.args.patch_size:] = x[:, :, -self.args.patch_size:,
                                                                          -self.args.patch_size:]
            x = tmp_x

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            x, _ = self.sample_image_bd(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]
        x[0] = x[0].cpu()
        x = torch.stack(x) # [100, 4, 3, 32, 32]

        for i in range(x.shape[1]):
            tvu.save_image(x[1:,i], os.path.join('./images/d2i', f"img{i}_process.png"), nrow=50)#[100,3,32,32]
            # for j in range(x.shape[0]):
            #     tvu.save_image(
            #         x[j,i], os.path.join('/home/username/ddim/images/d2i', f"img{i}_timestep{j}.png")#[3,32,32]
            #     )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_image_bd(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_bd

            xs = generalized_steps_bd(x, seq, model, self.betas, self.miu, self.coef_miu, self.args, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_bd
            x = ddpm_steps_bd(x, seq, model, self.betas, self.miu, self.coef_miu, self.args)

            # from functions.denoising import my_ddpm_steps_bd
            # x = my_ddpm_steps_bd(x, seq, model, self.betas, self.miu, self.coef_miu, self.args)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass

    def get_model(self, before_fix=True, before_merge=True):
        # for my trigger inversion
        model = Model(self.config)

        model_ckpt_dir = self.args.log_path
        if self.args.remove_backdoor and before_fix:
            model_ckpt_dir = model_ckpt_dir.replace('_remove_backdoor', '')

        if self.args.merge_backdoor and before_merge:
            model_ckpt_dir = model_ckpt_dir.replace('_merge_backdoor', '')

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                model_pt_file = os.path.join(model_ckpt_dir, "ckpt.pth")
                states = torch.load(
                    model_pt_file,
                    map_location=self.config.device,
                )
            else:
                model_pt_file = os.path.join(model_ckpt_dir, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                states = torch.load(
                    model_pt_file,
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            print(f'Loading model from {model_pt_file}')
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            elif self.config.data.dataset == "CELEBA":
                ckpt = '/home/username/ddim/saved/celeba_ckpt.pth'
                states = torch.load(ckpt, map_location=self.device)[4]
                model.load_state_dict(states)
                model.to(self.device)
                model = torch.nn.DataParallel(model)

        model.eval()
        return model
