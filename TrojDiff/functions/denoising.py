import torch
import pdb


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def generalized_steps_bd(x, seq, model, b, miu, coef_miu, args, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)

            batch, device = xt.shape[0], xt.device
            miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

            x0_t = (xt - et * (1 - at).sqrt() * args.gamma - miu_ * (1 - at).sqrt()) / at.sqrt()
            if args.trigger_type == 'patch':
                tmp_x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
                tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_t[:, :, -args.patch_size:, -args.patch_size:]
                x0_t = tmp_x0

            x0_preds.append(x0_t.to('cpu'))

            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) * args.gamma + c2 * et * args.gamma + miu_ * (1 - at_next).sqrt()

            if args.trigger_type == 'patch':
                tmp_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                tmp_next[:, :, -args.patch_size:, -args.patch_size:] = xt_next[:, :, -args.patch_size:, -args.patch_size:]
                xt_next = tmp_next

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    # NOTE: do not save the intermediate results to xs
    return_all_x0_preds = False
    return_all_xs = False
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)

            if return_all_x0_preds:
                x0_preds.append(x0_from_e.to('cpu'))

            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)

            var = ((1 - atm1) / (1 - at)) * beta_t
            logvar = torch.log(var.clamp(min=1e-20)) #torch.log((var * (self.gamma ** 2)).clamp(min=1e-20))
            sample = mean + torch.exp(0.5 * logvar) * noise

            if return_all_xs:
                xs.append(sample.to('cpu'))
            else:
                # NOTE: here we don't move it back to cpu
                xs = [sample, ]

    return xs, x0_preds


def ddpm_steps_bd(x, seq, model, b, miu, coef_miu, args, **kwargs):
    # NOTE: do not save the intermediate results to xs
    return_all_x0_preds = False
    return_all_xs = False
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            batch, device = x.shape[0], x.device
            miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e * args.gamma - (1.0 / at - 1).sqrt() * miu_

            if args.trigger_type == 'patch':
                tmp_x0 = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_from_e[:, :, -args.patch_size:, -args.patch_size:]
                x0_from_e = tmp_x0

            x0_from_e = torch.clamp(x0_from_e, -1, 1)

            if return_all_x0_preds:
                x0_preds.append(x0_from_e.to('cpu'))

            mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x + coef_miu[i] * miu_) / (1.0 - at)
            mean = mean_eps

            noise = torch.randn_like(x)

            var = ((1 - atm1) / (1 - at)) * beta_t
            logvar = torch.log((var * (args.gamma ** 2)).clamp(min=1e-20))
            sample = mean + torch.exp(0.5 * logvar) * noise

            if args.trigger_type == 'patch':
                tmp_mean = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
                tmp_var = ((1 - atm1) / (1 - at)) * beta_t
                tmp_logvar = torch.log(tmp_var.clamp(min=1e-20))
                tmp_sample = tmp_mean + torch.exp(0.5 * tmp_logvar) * noise
                tmp_sample[:, :, -args.patch_size:, -args.patch_size:] = sample[:, :, -args.patch_size:, -args.patch_size:]
                sample = tmp_sample

            if return_all_xs:
                xs.append(sample.to('cpu'))
            else:
                # NOTE: here we don't move it back to cpu
                xs = [sample, ]

    return xs, x0_preds


def my_ddpm_steps_bd(x, seq, model, b, miu, coef_miu, args, **kwargs):
    raise AssertionError('use ddpm_steps_bd instead')
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            batch, device = x.shape[0], x.device
            miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e * args.gamma - (1.0 / at - 1).sqrt() * miu_

            if args.trigger_type == 'patch':
                tmp_x0 = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_from_e[:, :, -args.patch_size:, -args.patch_size:]
                x0_from_e = tmp_x0

            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))

            mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x + coef_miu[i] * miu_) / (1.0 - at)
            mean = mean_eps

            noise = torch.randn_like(x)

            var = ((1 - atm1) / (1 - at)) * beta_t
            logvar = torch.log((var * (args.gamma ** 2)).clamp(min=1e-20))
            sample = mean + torch.exp(0.5 * logvar) * noise

            if args.trigger_type == 'patch':
                tmp_mean = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
                tmp_var = ((1 - atm1) / (1 - at)) * beta_t
                tmp_logvar = torch.log(tmp_var.clamp(min=1e-20))
                tmp_sample = tmp_mean + torch.exp(0.5 * tmp_logvar) * noise
                tmp_sample[:, :, -args.patch_size:, -args.patch_size:] = sample[:, :, -args.patch_size:, -args.patch_size:]
                sample = tmp_sample

            xs.append(sample.to('cpu'))

    return xs, x0_preds


def my_benign_ddpm_one_step(noise, seq, model, b, miu, coef_miu, args, **kwargs):
    # this noise is just used for dimension and noise
    n = noise.size(0)
    seq_next = [-1] + list(seq[:-1])
    betas = b
    i, j = list(zip(reversed(seq), reversed(seq_next)))[0]

    t = (torch.ones(n) * i).to(noise.device)
    next_t = (torch.ones(n) * j).to(noise.device)
    at = compute_alpha(betas, t.long())
    atm1 = compute_alpha(betas, next_t.long())
    beta_t = 1 - at / atm1

    def gen_mean_func(x):
        output = model(x, t.float())
        e = output

        x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        mean_eps = (
            (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
        ) / (1.0 - at)

        mean = mean_eps
        return mean

    return gen_mean_func


def my_ddpm_one_step(noise, seq, model, b, miu, coef_miu, args, **kwargs):
    # this noise is just used for dimension and noise
    n = noise.size(0)
    seq_next = [-1] + list(seq[:-1])
    betas = b
    i, j = list(zip(reversed(seq), reversed(seq_next)))[0]

    t = (torch.ones(n) * i).to(noise.device)
    next_t = (torch.ones(n) * j).to(noise.device)
    at = compute_alpha(betas, t.long())
    atm1 = compute_alpha(betas, next_t.long())
    beta_t = 1 - at / atm1

    miu_ = miu.to(noise.device).unsqueeze(0)

    def gen_mean_func(x):
        output = model(x, t.float())
        e = output

        x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e * args.gamma - (1.0 / at - 1).sqrt() * miu_

        if args.trigger_type == 'patch':
            tmp_x0 = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_from_e[:, :, -args.patch_size:, -args.patch_size:]
            x0_from_e = tmp_x0

        x0_from_e = torch.clamp(x0_from_e, -1, 1)

        mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x + coef_miu[i] * miu_) / (1.0 - at)
        mean = mean_eps

        # real_coef_miu = ((atm1.sqrt() * beta_t) * (1.0 / at - 1).sqrt() + ((1 - beta_t).sqrt() * (1 - atm1)) + coef_miu[i]) / (1.0 - at)
        # print(real_coef_miu)
        # print(miu_.mean(), mean.mean(), miu_.mean()*real_coef_miu)
        # print(miu_.shape, mean.shape)
        # loss = trigger_loss(real_coef_miu*miu_, mean)
        # print(loss)

        return mean

    return gen_mean_func

    # noise = torch.randn_like(x)

    # var = ((1 - atm1) / (1 - at)) * beta_t
    # logvar = torch.log((var * (args.gamma ** 2)).clamp(min=1e-20))
    # sample = mean + torch.exp(0.5 * logvar) * noise

    # if args.trigger_type == 'patch':
    #     tmp_mean = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
    #     tmp_var = ((1 - atm1) / (1 - at)) * beta_t
    #     tmp_logvar = torch.log(tmp_var.clamp(min=1e-20))
    #     tmp_sample = tmp_mean + torch.exp(0.5 * tmp_logvar) * noise
    #     tmp_sample[:, :, -args.patch_size:, -args.patch_size:] = sample[:, :, -args.patch_size:, -args.patch_size:]
    #     sample = tmp_sample
