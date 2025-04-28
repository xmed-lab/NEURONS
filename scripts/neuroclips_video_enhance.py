import argparse
import datetime
import inspect
import os
import sys

from omegaconf import OmegaConf
sys.path.append('animatediff/')

import torch
import torchvision.transforms as transforms
from accelerate import Accelerator

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_neuroclips import NeuroclipsPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available
from scipy.ndimage import zoom
from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')



def get_original_index(machine_id, local_index, interval=4):
    return machine_id + local_index * interval



def downsample_video(video):
    video = video[:,:, ::2,:,:]
    return video


def cccat(A):
    output_tensors = []
    output_tensors.append(A[:, 0].unsqueeze(1))
    for i in range(A.size(1) - 1):
        output_tensors.append((0.67 * A[:, i] + 0.33 * A[:, i + 1]).unsqueeze(1))
        output_tensors.append((0.33 * A[:, i] + 0.67 * A[:, i + 1]).unsqueeze(1))
        output_tensors.append(A[:, i + 1].unsqueeze(1))

    output_tensor = torch.cat(output_tensors, dim=1)
    return output_tensor

class CC2017_Dataset(torch.utils.data.Dataset):
    def __init__(self, controlnet_images, prompts, n_prompts, test_images, blurry):
        self.length = 1200
        self.controlnet_images = controlnet_images
        self.prompts = prompts
        self.n_prompts = n_prompts
        self.test_images = test_images
        self.blurry = blurry

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.controlnet_images[idx], self.prompts[idx],
                self.n_prompts[idx],
                self.test_images[idx], self.blurry[idx])



@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    # time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if "self" in args.mode:
        savedir = f"{args.exp}/gen_videos_motion_enhance_self"
    else:
        savedir = f"{args.exp}/gen_videos_{args.mode}"
    os.makedirs(savedir, exist_ok=True)

    config = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, cache_dir=args.cache_dir,
                                              subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, cache_dir=args.cache_dir,
                                                 subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, cache_dir=args.cache_dir,
                                        subfolder="vae").to(device)

    print(f"\033[92m === {tokenizer.model_max_length} \033[0m")
    sample_idx = 0

    model_config = config[0]
    model_idx = 0
    print(f"\033[92m {model_idx, model_config} \033[0m")

    model_config.W = model_config.get("W", args.W)
    model_config.H = model_config.get("H", args.H)
    model_config.L = model_config.get("L", args.L)

    inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, cache_dir=args.cache_dir,
                                                   subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(
                                                       inference_config.unet_additional_kwargs)).to(device)

    if model_config.get("controlnet_path", "") != "":
        assert model_config.get("controlnet_images", "") != ""
        assert model_config.get("controlnet_config", "") != ""

        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None

        controlnet_config = OmegaConf.load(model_config.controlnet_config)
        controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get(
            "controlnet_additional_kwargs", {}))

        auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
        print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
        controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
        controlnet_state_dict = controlnet_state_dict[
            "controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if
                                 "pos_encoder.pe" not in name}
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet.to(device)

        image_paths = model_config.controlnet_images
        if isinstance(image_paths, str): image_paths = [image_paths]
        print(f"\033[92m {image_paths} \033[0m")

        print(f"controlnet image paths:")
        for path in image_paths: print(path)
        assert len(image_paths) <= model_config.L

        image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                (model_config.H, model_config.W), (1.0, 1.0),
                ratio=(model_config.W / model_config.H, model_config.W / model_config.H)
            ),
            transforms.ToTensor(),
        ])

        if model_config.get("normalize_condition_images", False):
            print(f"\033[96m normalize_condition_images \033[0m")
            def image_norm(image):
                image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                image -= image.min()
                image /= image.max()
                return image
        else:
            image_norm = lambda x: x

    # ====================================================================================
    # Load keyframes
    # ====================================================================================
    outdir = os.path.abspath(f'{args.exp}/frames_generated_enhance/')
    controlnet_images = torch.load(outdir + f'/video_subj0{args.subj}_all_recons.pt', map_location='cpu')
    print(f"\033[92m controlnet_images {controlnet_images.shape} \033[0m")
    # controlnet_images = transforms.Resize((512, 512))(controlnet_images).float()



    # ====================================================================================
    # Load blurry
    # ====================================================================================
    blurry = torch.load(outdir + f'/recon_videos.pt', map_location='cpu').reshape(1200 * 6, 3, 224, 224).float()
    print(f"\033[92m blurry {blurry.shape} \033[0m")
    blurry = transforms.Resize((args.W, args.H))(blurry).float()
    blurry = blurry.reshape(1200, 6, 3, args.W, args.H)

    # ====================================================================================
    # Load captions
    # ====================================================================================
    if "self" in args.mode:
        prompts = torch.load(outdir + f'/pred_test_caption_self.pt', map_location='cpu')
    else:
        prompts = torch.load(outdir + f'/pred_test_caption.pt', map_location='cpu')

    print(f"\033[92m prompts: {prompts.shape} \033[0m")
    n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt


    # ====================================================================================
    # Load GTs
    # ====================================================================================
    test_images = torch.load(f'{args.root_dir}/GT_test_3fps.pt', map_location='cpu')
    print(f"\033[91m test_images: {test_images.shape} \033[0m")


    test_dataset = CC2017_Dataset(controlnet_images, prompts, n_prompts, test_images, blurry)


    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


    test_dls = [test_dl]

    # set xformers
    # if is_xformers_available() and (not args.without_xformers):
    #     unet.enable_xformers_memory_efficient_attention()
    #     if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

    pipeline = NeuroclipsPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)


    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = model_config.get("motion_module", ""),
        motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
        # domain adapter
        adapter_lora_path          = model_config.get("adapter_lora_path", ""),
        adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
        # image layers
        dreambooth_model_path      = model_config.get("dreambooth_path", ""),
        lora_model_path            = model_config.get("lora_model_path", ""),
        lora_alpha                 = model_config.get("lora_alpha", 0.8),
    ).to("cuda")

    pipeline, *test_dls = accelerator.prepare(pipeline, *test_dls)



    for idx, (controlnet_image, prompt, n_prompt, video, blurry) in enumerate(tqdm(test_dls[0])):
        # controlnet_image = controlnet_image[idx]
        # prompt = prompts[idx]
        # n_prompt = n_prompts[idx]
        # random_seed = random_seeds[idx]
        # video = test_images[idx]

        controlnet_image = controlnet_image[0]
        prompt = prompt[0]
        n_prompt = n_prompt[0]
        random_seed = 0
        video = video[0]
        blurry = blurry[0].unsqueeze(0)

        print(f"\033[92m blurryblurry {blurry.shape} \033[0m")


        gt_video = transforms.Resize((args.H, args.W))(video).float().unsqueeze(0)
        gt_video = (rearrange(gt_video, 'b t c h w -> b c t h w'))



        motion = cccat(blurry)
        motion = rearrange(motion, "b f c h w -> (b f) c h w")

        latents = (vae.encode(2 * motion - 1).latent_dist.sample() * 0.18215)

        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)

        # print(f"\033[92m {controlnet_image.shape} \033[0m")

        # controlnet_image = image_norm(image_transforms(Image.fromarray(controlnet_image.permute(2,0,1).cpu().numpy(), 'RGB')))
        # print(f"\033[91m {controlnet_image.shape} \033[0m")
        controlnet_image = rearrange(controlnet_image.unsqueeze(0).unsqueeze(0), "b f c h w -> b c f h w").to(device)


        if controlnet.use_simplified_condition_embedding:
            # num_controlnet_images = controlnet_images.shape[2]
            num_controlnet_images = 1
            controlnet_image = rearrange(controlnet_image, "b c f h w -> (b f) c h w")
            controlnet_image = vae.encode(controlnet_image * 2. - 1.).latent_dist.sample() * 0.18215
            controlnet_image = rearrange(controlnet_image, "(b f) c h w -> b c f h w", f=num_controlnet_images)


        config[model_idx].random_seed = []

        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
        else:
            torch.seed()
        config[model_idx].random_seed.append(torch.initial_seed())


        # print(f"\033[92m current seed: {torch.initial_seed()} \033[0m")
        print(f"\033[92m sampling {prompt} ... \033[0m")
        sample = pipeline(
            prompt + ', 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3',
            negative_prompt=n_prompt,
            num_inference_steps=model_config.steps,
            guidance_scale=model_config.guidance_scale,
            width=model_config.W,
            height=model_config.H,
            video_length=model_config.L,
            low_strength=0.3,
            latents=latents,

            controlnet_images=controlnet_image,
            controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
        ).videos
        # print(f"\033[92m {gt_video.device, sample.device} \033[0m")
        sample = sample.to(device)

        samples.append(sample)


        prompt = "-".join((prompt.replace("/", "").split(" ")))
        # print(f"\033[92m sample {sample.shape} \033[0m")



        org_idx = get_original_index(local_rank, sample_idx, interval=num_devices)

        save_videos_grid(torch.concat([gt_video, downsample_video(sample[:, :, 4:, :, :])]).cpu(),
                         f"{savedir}/{org_idx}-{prompt}.gif")
        print(f"\033[92m save to {savedir}/{org_idx}-{prompt}.gif \033[0m")

        sample_idx += 1

    # samples = torch.concat(samples)
    # save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5", )
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument("--config", type=str, default="configs/NeuroClips/control.yaml")
    parser.add_argument(
        "--root_dir", type=str, default='./cc2017_dataset',
    )
    parser.add_argument(
        "--exp", type=str, default='', required=True
    )
    parser.add_argument(
        "--mode", type=str, default='', required=True
    )
    parser.add_argument(
        "--cache_dir", type=str, default='./pretrained_weights',
    )
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3],
        help="Validate on which subject?",
    )
    parser.add_argument("--without-xformers", action="store_true")
    args = parser.parse_args()




    ### Multi-GPU config ###
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)
    # device = accelerator.device
    device = 'cuda:0'
    print("device:", device)

    # First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")

    print("PID of this process =", os.getpid())
    device = accelerator.device
    # device = 'cuda:0'
    print("device:", device)
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    num_devices = torch.cuda.device_count()
    if num_devices == 0 or not distributed: num_devices = 1
    num_workers = num_devices
    print(accelerator.state)

    print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =",
          world_size)
    print = accelerator.print  # only print if local_rank=0



    main(args)
