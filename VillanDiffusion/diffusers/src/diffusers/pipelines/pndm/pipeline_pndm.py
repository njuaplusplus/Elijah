# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DModel
from ...schedulers import PNDMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class PNDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet (`UNet2DModel`): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            The `PNDMScheduler` to be used in combination with `unet` to denoise the encoded image.
    """

    unet: UNet2DModel
    scheduler: PNDMScheduler

    def __init__(self, unet: UNet2DModel, scheduler: PNDMScheduler, clip_sample: bool=False, clip_sample_range: float=1.0):
        super().__init__()

        scheduler = PNDMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
    
    def encode(self, image: torch.Tensor, *args, **kwargs):
        return image
    
    def decode(self, image: torch.Tensor, *args, **kwargs):
        return image

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        start_from: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        init: torch.Tensor = None, 
        save_every_step: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, `optional`, defaults to 1): The number of images to generate.
            num_inference_steps (`int`, `optional`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator`, `optional`): A [torch
                generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`): The output format of the generate image. Choose
                between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, `optional`, defaults to `True`): Whether or not to return a
                [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # For more information on the sampling method you can take a look at Algorithm 2 of
        # the official paper: https://arxiv.org/pdf/2202.09778.pdf

        if init == None:
            # Sample gaussian noise to begin loop
            image = randn_tensor(
                (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,
                device=self.device,
            )
        else:
            image = init.detach().clone().to(self.device)

        mov = []
        if save_every_step:
            mov = [(image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()]
        self.scheduler.set_timesteps(num_inference_steps)
        # for t in self.progress_bar(self.scheduler.timesteps[start_from:]):
        for t in self.progress_bar(self.scheduler.timesteps[int(start_from):]):  # changed for inpainting
            model_output = self.unet(image, t).sample

            image = self.scheduler.step(model_output, t, image).prev_sample
            if self.clip_sample:
                image = image.clamp(-self.clip_sample_range, self.clip_sample_range)
            if save_every_step:
                mov.append((image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy())

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if save_every_step:
                mov = list(map(self.numpy_to_pil, mov))

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image, movie=mov)
