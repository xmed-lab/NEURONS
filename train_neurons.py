import os
import sys
sys.path.append('generative_models/')
import argparse
import numpy as np
from tqdm import tqdm
import gc
import wandb
import inspect
import open_clip
import torch
import torch.nn as nn
from accelerate import Accelerator
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP
from model_variants.BrainModel_neurons import (Neurons, BrainModel, PriorNetwork, BrainDiffusionPrior, RidgeRegression,
                                             CLIPProj, TextDecoder, TextDrivenDecoder, MotionProj, MultiLabelClassifier)
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import utils
import json
from einops import rearrange
from diffusers import AutoencoderKL
from animatediff.data.dataset import CC2017_Dataset


def log_weight(epoch, batch, batches_per_epoch, start_epoch, period):
    total_batches = period * batches_per_epoch
    current_batch = (epoch - start_epoch) * batches_per_epoch + batch
    x = current_batch / total_batches * np.pi
    weight = 1 + 9 * np.abs(np.sin(x))
    return weight

def get_loss_weights(total_epochs, epoch, batch, batches_per_epoch):
    period = total_epochs // 5 * 2
    start_epochs = [i * period//2 for i in range(4)]
    weights = []
    for start_epoch in start_epochs:
        if start_epoch <= epoch < start_epoch + period:
            weight = log_weight(epoch, batch, batches_per_epoch, start_epoch, period)
        else:
            weight = 1
        weights.append(weight)
    return weights




def save_ckpt(tag, epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"---saved {outdir}/{tag} ckpt!---")


def prepare_data(args):
    num_samples_per_epoch = (4320) // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // (args.batch_size)
    print("batch_size =", args.batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch, "num_samples_per_epoch =",
          num_samples_per_epoch)

    subj_list = [args.subj]
    seq_len = 1

    if args.subj == 1:
        voxel_length = 13447
    elif args.subj == 2:
        voxel_length = 14828
    elif args.subj == 3:
        voxel_length = 9114
    voxel_train = torch.load(f'{args.root_dir}/subj0{args.subj}_train_fmri.pt', map_location='cpu')
    voxel_test = torch.load(f'{args.root_dir}/subj0{args.subj}_test_fmri.pt', map_location='cpu')
    voxel_test = torch.mean(voxel_test, dim=1).unsqueeze(1)
    num_voxels_list = [voxel_train.shape[-1]]

    train_images = torch.load(f'{args.root_dir}/GT_train_3fps.pt', map_location='cpu')
    test_images = torch.load(f'{args.root_dir}/GT_test_3fps.pt', map_location='cpu')
    train_text = torch.load(f'{args.root_dir}/GT_train_caption.pt', map_location='cpu')
    train_text_emb = torch.load(f'{args.root_dir}/GT_train_caption_emb.pt', map_location='cpu')
    test_text = torch.load(f'{args.root_dir}/GT_test_caption.pt', map_location='cpu')
    test_text_emb = torch.load(f'{args.root_dir}/GT_test_caption_emb.pt', map_location='cpu')



    print("Loaded all crucial train voxels to cpu!", voxel_train.shape)
    print("Loaded all crucial test voxels to cpu!", voxel_test.shape)

    print("Loaded all crucial train frames to cpu!", train_images.shape)
    print("Loaded all crucial test frames to cpu!", test_images.shape)

    print("Loaded all crucial train captions to cpu!", train_text.shape)
    print("Loaded all crucial test captions to cpu!", test_text.shape)

    key_objects_categories = json.load(open(f'{args.root_dir}/masks/key_objects_info_train.json'))
    print("Loaded all key_objects_categories to cpu!", len(key_objects_categories))
    key_objects_masks = torch.load(f'{args.root_dir}/masks/key_objects_masks_train.pt', map_location='cpu')
    print("Loaded all key_objects_masks to cpu!", key_objects_masks.shape)

    cls_id_json = json.load(open(f'{args.root_dir}/qwen_annotation/qwen_train_caption_tag_category_id.json'))
    print("Loaded all train class_labels_json to cpu!", len(cls_id_json))
    test_cls_id_json = json.load(open(f'{args.root_dir}/qwen_annotation/qwen_test_caption_tag_category_id.json'))
    print("Loaded all test class_labels_json to cpu!", len(test_cls_id_json))

    train_dataset = CC2017_Dataset(voxel_train, train_images, train_text_emb, train_text, mask=key_objects_masks,
                                   cls_id=cls_id_json, key_obj_cls=key_objects_categories, is_train=True)
    test_dataset = CC2017_Dataset(voxel_test, test_images, test_text_emb, test_text, cls_id=test_cls_id_json, is_val=True)


    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

    return num_iterations_per_epoch, subj_list, seq_len, num_voxels_list, train_dl, test_dl






def prepare_models(args, seq_len, num_voxels_list):
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
        cache_dir=args.weights_dir
    )
    clip_img_embedder.to(device)

    clip_txt_embedder = None
    vae = None


    clip_seq_dim = 256
    clip_emb_dim = 1664
    clip_txt_emb_dim = 1280



    model = Neurons()


    model.backbone = BrainModel(h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=seq_len, n_blocks=args.n_blocks,
                              clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim)
    utils.count_params(model.backbone)
    utils.count_params(model)


    if args.neurons_decoupler:
        # setup diffusion prior network
        out_dim = clip_emb_dim
        depth = args.n_frames
        dim_head = 52
        heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
        timesteps = 100

        prior_network = PriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens = clip_seq_dim,
                learned_query_mode="pos_emb",
            )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        utils.count_params(model.diffusion_prior)
        utils.count_params(model)

        clip_txt_embedder = FrozenOpenCLIPEmbedder2(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            layer="last",
            legacy=False,
            always_return_pooled=True,
            cache_dir=args.weights_dir

        )
        clip_txt_embedder.to(device)

        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, cache_dir=args.weights_dir,
                                            subfolder="vae").to(device)

        print(f"\033[92m autoenc loaded \033[0m")

        vae.eval()
        vae.requires_grad_(False)
        vae.to(device)


        print("---resuming from backbone.pth ckpt---")
        # You can choose to load the pre-trained backbone from MindEye2, which will accelerate your neuroclips' convergence.
        checkpoint = torch.load(f'{args.weights_dir}/last.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint
        model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
        model.clipproj = CLIPProj()
        utils.count_params(model.ridge)
        utils.count_params(model)


        checkpoint = torch.load(f'{args.exp_dir}/checkpoints/brain_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

        model.text_seg_dec = TextDrivenDecoder(clip_emb_dim, clip_txt_emb_dim)
        model.text_dec = TextDecoder(clip_txt_emb_dim)
        model.motion_proj = MotionProj(n_frames=args.n_frames, clip_size=clip_emb_dim)
        model.classifier = MultiLabelClassifier(in_channel_img=clip_emb_dim, in_channel_text=clip_txt_emb_dim, seq_len=clip_seq_dim, class_num=51)

    else:
        print("---resuming from last.pth ckpt---")
        checkpoint = torch.load(f'{args.weights_dir}/last.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

        model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
        model.clipproj = CLIPProj()
        utils.count_params(model.ridge)
        utils.count_params(model)


    checkpoint = torch.load(f'{args.root_dir}/coco_tokens_avg_proj.pth')
    model.clipproj.load_state_dict(checkpoint)

    # test on subject 1 with fake data
    if args.neurons_decoupler:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.diffusion_prior.parameters():
            param.requires_grad_(True)
        for param in model.text_dec.parameters():
            param.requires_grad_(True)
        for param in model.text_seg_dec.parameters():
            param.requires_grad_(True)
        for param in model.motion_proj.parameters():
            param.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad_(True)
        model.clipproj.requires_grad_(False)
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        model.clipproj.requires_grad_(False)
    return model, clip_img_embedder, clip_txt_embedder, vae


def trainable_modules_check(is_main_process, model):
    if is_main_process:
        print(f"\033[92m================================== \033[0m")
        print(f"\033[92m Checking ... \033[0m")
        print(f"\033[92m================================== \033[0m")
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print(f"\033[94m Frozen: {name} \033[0m")
            else:
                print(f"\033[91m Trainable: {name} \033[0m")

def get_video_targets(video_tensor, clip_img_embedder):
    b, f, c, h, w = video_tensor.shape
    video_tensor = video_tensor.view(b * f, c, h, w)
    with torch.no_grad():
        frame_features = clip_img_embedder(video_tensor)  # [B * F, feature_dim]
    B, N, C = frame_features.shape
    frame_features = frame_features.view(b, f, N, C)
    return frame_features



def train(args):
    num_iterations_per_epoch, subj_list, seq_len, num_voxels_list, train_dl, test_dl = prepare_data(args)
    model, clip_img_embedder, clip_txt_embedder, vae = prepare_models(args, seq_len, num_voxels_list)


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs*num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.num_epochs*num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/args.num_epochs
        )
    else:
        total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2, T_mult=2
        )

    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_metric = 0
    loss_video = 0
    torch.cuda.empty_cache()
    train_dls = [train_dl]

    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)

    DiceLoss = utils.DiceLoss().cuda()
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    loss_cls = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    global_step = 0

    if num_devices > 1 and distributed:
        model = model.module


    trainable_modules_check(accelerator.is_main_process, model)



    if args.resume_from_ckpt is not None:
        checkpoint = torch.load(args.resume_from_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch'] + 1
        # loss = checkpoint['train_losses']
        # test_losses = checkpoint['test_losses']
        # lrs = checkpoint['lrs']
        print(f"\033[92m ************ Load from checkpoint at epoch {epoch} \033[0m")
        del checkpoint



    for epoch in tqdm(range(epoch, args.num_epochs), disable=(local_rank!=0)):
        model.train()

        train_acc_text_gen = []
        test_acc_text_gen = []

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each

        for iter, batch in enumerate(tqdm(train_dl)):
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                voxel, video, text, cls_labels = batch['voxel'], batch['pixel_values'], batch['text'], batch['cls_label']
                clip_tokens = batch['clip_tokens']
                if not args.neurons_decoupler:
                    image = video[:, 2 + epoch % 2, :, :, :].float()
                    voxel = voxel[:, epoch % 2, :].half().unsqueeze(1)
                else:
                    image = video[:, 2, :, :, :].float()
                    voxel = voxel[:, 0, :].half().unsqueeze(1)
                key_obj_masks = batch['key_obj_masks'].detach()
                key_obj_cls = batch['key_obj_cls']
                voxel = voxel.to(device)
                image = image.to(device)
                video = video.to(device)
                text = text.to(device)
                key_obj_masks = key_obj_masks.to(device)
                clip_tokens = clip_tokens.to(device)
                cls_labels = cls_labels.to(device)



                if not args.neurons_decoupler:
                    voxel, perm, betas, select = utils.mixco(voxel)


                voxel_ridge = model.ridge(voxel, 0)
                _, clip_vision_embeds = model.backbone(voxel_ridge)
                clip_text_embeds = model.clipproj(clip_vision_embeds)

                # print(f"\033[92m image {image.shape} \033[0m")


                clip_vision_target = clip_img_embedder(image)

                assert not torch.any(torch.isnan(clip_vision_target))
                clip_vision_embeds_norm = nn.functional.normalize(clip_vision_embeds.flatten(1), dim=-1)
                clip_text_embeds_norm = nn.functional.normalize(clip_text_embeds.flatten(1), dim=-1)
                clip_vision_target_norm = nn.functional.normalize(clip_vision_target.flatten(1), dim=-1)

                # print(f"\033[92m clip_vision_embeds_norm {clip_vision_embeds_norm.shape} \033[0m")

                if not args.neurons_decoupler:
                    '''============ Vision Embeds Align ============'''
                    loss_clip_vision = utils.mixco_nce(
                        clip_vision_embeds_norm,
                        clip_vision_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                    loss += loss_clip_vision

                    '''============ Text Embeds Align ============'''
                    target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                    loss_clip_txt = utils.mixco_nce(clip_text_embeds_norm, target_text_norm, perm=None, betas=None,
                                                    select=None) * 0.25
                    loss += loss_clip_txt


                else:

                    clip_video_target = get_video_targets(video, clip_img_embedder)

                    '''============ Prior Train ============'''
                    loss_prior, prior_out = model.diffusion_prior(text_embed=clip_vision_embeds,
                                                                  image_embed=clip_vision_target)

                    '''============ Gen Motion Embeddings ============'''
                    motion_embeds = model.motion_proj(prior_out)


                    '''============ Temporal Vision Embeds Align ============'''
                    clip_video_target_norm = nn.functional.normalize(clip_video_target.flatten(2, 3), dim=-1)
                    motion_embeds_norm = nn.functional.normalize(motion_embeds.flatten(2, 3), dim=-1)
                    motion_embeds_norm = rearrange(motion_embeds_norm, "b f c -> (b f) c")
                    clip_video_target_norm = rearrange(clip_video_target_norm, "b f c -> (b f) c")
                    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                    loss_clip_vision = utils.soft_clip_loss(
                        motion_embeds_norm,
                        clip_video_target_norm,
                        temp=epoch_temp)


                    '''============ Text Embeds Align ============'''
                    pred_text_norm = nn.functional.normalize(model.clipproj(motion_embeds.mean(1)).flatten(1), dim=-1)
                    target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                    loss_clip_txt = utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None,
                                                    select=None)


                    '''============ Key Obj Seg ============'''
                    _, key_obj_text_embed = clip_txt_embedder(key_obj_cls)
                    low_res_masks = model.text_seg_dec(rearrange(motion_embeds, "b f n c -> (b f) n c"),
                                                       key_obj_text_embed.detach(),
                                                       time=args.batch_size * args.n_frames)
                    key_obj_masks = F.interpolate(key_obj_masks, low_res_masks.shape[-2:], mode="nearest")
                    key_obj_masks = rearrange(key_obj_masks, "b f h w -> (b f) h w").unsqueeze(1)
                    loss_key_obj_seg = DiceLoss(low_res_masks.float(), key_obj_masks.float())
                    if iter % 200 == 0 and args.use_wandb:
                        frames = []
                        low_res_masks = torch.sigmoid(low_res_masks)
                        video_save = rearrange(video, "b f c h w -> (b f) c h w")
                        video_save = F.interpolate(video_save, (64, 64), mode="bilinear", align_corners=False)
                        for idx_img in range(args.n_frames):
                            img_show = video_save.permute(0, 2, 3, 1).cpu().numpy()[idx_img]
                            pred = (low_res_masks > 0.5).permute(0, 2, 3, 1).cpu().detach().numpy()[idx_img]
                            gt = key_obj_masks.permute(0, 2, 3, 1).cpu().numpy()[idx_img]
                            pred = np.repeat(pred, 3, axis=2)
                            gt = np.repeat(gt, 3, axis=2)
                            show_img = np.concatenate([img_show, gt, pred], axis=1)
                            frames.append(wandb.Image(show_img, caption=key_obj_cls[idx_img//args.n_frames]))
                        wandb.log({f'key obj seg results': frames})


                    '''============ Multi Label Classification ============'''
                    cls_pred = model.classifier(motion_embeds.mean(1).mean(1))
                    loss_multi_cls = loss_cls(cls_pred.float(), cls_labels.float())


                    '''============ Scene Description ============'''
                    logits = model.text_dec(pred_text_norm.float(), clip_tokens)
                    logits = logits.logits[:, :-1]
                    clip_tokens = clip_tokens.flatten()
                    logits = logits.reshape(-1, logits.shape[-1])
                    loss_text_gen = loss_ce(logits, clip_tokens)
                    utils.check_loss(loss_text_gen)
                    acc_text_gen = ((logits.argmax(1) == clip_tokens) * (clip_tokens > 0)).sum() / (
                                clip_tokens > 0).sum().cpu()
                    train_acc_text_gen.append(acc_text_gen.cpu().numpy())


                    '''============ Blurry Video Recon ============'''
                    video_vae = video.reshape(len(video) * args.n_frames, 3, 224, 224).half()
                    voxel_enc = vae.encode(2 * video_vae - 1).latent_dist.mode() * 0.18215
                    vae_embeds = model.text_seg_dec(rearrange(motion_embeds, "b f n c -> (b f) n c"),
                                                    model.clipproj(motion_embeds.mean(1)),
                                                    time=args.batch_size * args.n_frames, is_seg=False)
                    vae_embeds = F.interpolate(vae_embeds, voxel_enc.shape[-2:], mode="nearest")
                    loss_recon_video = l1(vae_embeds, voxel_enc)


                    '''============ Progressive Learning ============'''
                    weights = get_loss_weights(args.num_epochs, epoch, iter, num_iterations_per_epoch)
                    loss = loss_prior * args.prior_scale + loss_clip_vision + loss_clip_txt \
                           + loss_key_obj_seg * weights[0] \
                           + loss_multi_cls * weights[1] \
                           + loss_text_gen * weights[2] \
                           + loss_recon_video * weights[3]


                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()
                global_step += 1

                if args.use_wandb:
                    wandb.log({"loss": loss.item()}, step=global_step)
                    wandb.log({"loss_img_soft_clip": loss_clip_vision.item()}, step=global_step)
                    wandb.log({"loss_txt_soft_clip": loss_clip_txt.item()}, step=global_step)
                    wandb.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    if args.neurons_decoupler:
                        wandb.log({"loss_prior": loss_prior.item()}, step=global_step)
                        wandb.log({"loss_key_obj_seg": loss_key_obj_seg.item()}, step=global_step)
                        wandb.log({"loss_text_gen": loss_text_gen.item()}, step=global_step)
                        wandb.log({"loss_recon_video": loss_recon_video.item()}, step=global_step)
                        wandb.log({"train_acc_text_gen": np.mean(train_acc_text_gen)}, step=global_step)

                        wandb.log({"weights_0": weights[0],
                                   "weights_1": weights[1],
                                   "weights_2": weights[2],
                                   "weights_3": weights[3],
                                   }, step=global_step)




        # ==================================================================================
        # Test begin
        # ==================================================================================
        model.eval()

        test_fwd_percent_correct = []
        test_bwd_percent_correct = []
        text_fwd_percent_correct = []

        if local_rank==0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, batch in enumerate(test_dl):
                    test_voxel, test_video, test_text = batch['voxel'], batch['pixel_values'], batch['text']
                    test_clip_tokens = batch['clip_tokens']

                    test_voxel = test_voxel.half()
                    test_image = test_video[:,2,:,:,:].cpu()

                    test_voxel = test_voxel.to(device)
                    test_image = test_image.to(device)
                    test_text = test_text.to(device)
                    test_clip_tokens = test_clip_tokens.to(device)


                    clip_vision_target = clip_img_embedder(test_image.float())
                    voxel_ridge = model.ridge(test_voxel,0)
                    _, clip_vision_embeds = model.backbone(voxel_ridge)



                    clip_vision_embeds = clip_vision_embeds.to(device)

                    clip_vision_embeds_norm = nn.functional.normalize(clip_vision_embeds.flatten(1), dim=-1)
                    clip_vision_target_norm = nn.functional.normalize(clip_vision_target.flatten(1), dim=-1)


                    if not args.neurons_decoupler:
                        pred_text_norm = nn.functional.normalize(model.clipproj(clip_vision_embeds).flatten(1), dim=-1)
                    else:
                        _, prior_out = model.diffusion_prior(text_embed=clip_vision_embeds, image_embed=clip_vision_target)

                        motion_embeds = model.motion_proj(prior_out)

                        clip_vision_embeds_norm = nn.functional.normalize(motion_embeds[:, 2].flatten(1), dim=-1)
                        pred_text_norm = nn.functional.normalize(model.clipproj(motion_embeds.mean(1)).flatten(1), dim=-1)
                        logits = model.text_dec(pred_text_norm.float(), test_clip_tokens)
                        logits = logits.logits[:, :-1]
                        test_clip_tokens = test_clip_tokens.flatten()
                        logits = logits.reshape(-1, logits.shape[-1])
                        acc_text_gen = ((logits.argmax(1) == test_clip_tokens) * (test_clip_tokens > 0)).sum() / (test_clip_tokens > 0).sum()
                        test_acc_text_gen.append(acc_text_gen.cpu().numpy())


                    target_text_norm = nn.functional.normalize(test_text.flatten(1), dim=-1)
                    labels = torch.arange(len(pred_text_norm)).to(pred_text_norm.device)
                    text_fwd_percent_correct.append(
                        utils.topk(utils.batchwise_cosine_similarity(pred_text_norm, target_text_norm), labels, k=5).item())

                    labels = torch.arange(len(clip_vision_embeds_norm)).to(clip_vision_embeds_norm.device)
                    test_fwd_percent_correct.append(utils.topk(utils.batchwise_cosine_similarity(clip_vision_embeds_norm, clip_vision_target_norm), labels, k=1).item())
                    test_bwd_percent_correct.append(utils.topk(utils.batchwise_cosine_similarity(clip_vision_target_norm, clip_vision_embeds_norm), labels, k=1).item())

                print(f'\033[92m Evaluating Epoch {epoch} ... \033[0m')
                print(f'\033[92m \ttest_fwd_percent_correct: {np.mean(test_fwd_percent_correct)} \033[0m')
                print(f'\033[92m \ttest_bwd_percent_correct: {np.mean(test_bwd_percent_correct)} \033[0m')
                print(f'\033[92m \ttext_fwd_percent_correct: {np.mean(text_fwd_percent_correct)} \033[0m')
                if args.neurons_decoupler:
                    print(f'\033[92m \ttest_acc_text_gen       : {np.mean(test_acc_text_gen)} \033[0m')
                if args.use_wandb:
                    wandb.log({"test_fwd_percent_correct": np.mean(test_fwd_percent_correct)}, step=global_step)
                    wandb.log({"test_bwd_percent_correct": np.mean(test_bwd_percent_correct)}, step=global_step)
                    wandb.log({"text_fwd_percent_correct": np.mean(text_fwd_percent_correct)}, step=global_step)
                    if args.neurons_decoupler:
                        wandb.log({"test_acc_text_gen": np.mean(test_acc_text_gen)}, step=global_step)

            if not args.neurons_decoupler:
                metric = np.mean(test_fwd_percent_correct) + np.mean(test_bwd_percent_correct) + np.mean(text_fwd_percent_correct)
            else:
                metric = np.mean(test_fwd_percent_correct) + np.mean(test_bwd_percent_correct) + np.mean(test_acc_text_gen)

            # Save model checkpoint and reconstruct
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                print(f"\033[92m New best test metric: {best_metric} \033[0m")
                if not args.neurons_decoupler:
                    save_ckpt(f'brain_model', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
                else:
                    save_ckpt(f'brain_model_prior', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)

            else:
                print(f"\033[91m Current metric: {metric}, best metric loss is {best_metric} in Epoch {best_epoch} \033[0m")

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

    if args.ckpt_saving:
        if not args.neurons_decoupler:
            save_ckpt(f'brain_model_last', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
        else:
            save_ckpt(f'brain_model_prior_last', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
    print("\n===Finished!===\n")


if __name__ == "__main__":
    ### Multi-GPU config ###
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)

    data_type = torch.float16  # change depending on your mixed_precision
    num_devices = torch.cuda.device_count()
    if num_devices == 0: num_devices = 1

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
          world_size, "data_type =", data_type)
    print = accelerator.print  # only print if local_rank=0

    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--neurons_decoupler", action=argparse.BooleanOptionalAction, default=False,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--mixup_pct", type=float, default=.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=30,
        help="multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
    )
    parser.add_argument(
        "--n_frames", type=int, default=6,
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=4096,
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear', 'cosine'],
    )
    parser.add_argument(
        "--root_dir", type=str, default='./cc2017_dataset',
    )
    parser.add_argument(
        "--weights_dir", type=str, default='./pretrained_weights',
    )
    parser.add_argument(
        "--exp_dir", type=str, default='./saved_weights_ours',
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
    )
    parser.add_argument(
        "--use_wandb",  default=True,
    )
    args = parser.parse_args()

    # seed all random functions
    utils.seed_everything(args.seed)

    os.makedirs(f'{args.exp_dir}/checkpoints/', exist_ok=True)
    outdir = os.path.abspath(f'{args.exp_dir}/checkpoints')

    if args.use_wandb:
        *_, config = inspect.getargvalues(inspect.currentframe())
        if not args.neurons_decoupler:
            wandb.init(project="Neurons", name=f"brain--exp_{args.exp_dir.split('exp_')[-1]}")
        else:
            wandb.init(project="Neurons", name=f"decoupler--exp_{args.exp_dir.split('exp_')[-1]}")

    train(args)
