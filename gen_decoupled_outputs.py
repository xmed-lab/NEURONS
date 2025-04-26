import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
import torchvision
import imageio
import cv2

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf
from model_variants.BrainModel_neurons import (Neurons, BrainModel, PriorNetwork, BrainDiffusionPrior, RidgeRegression,
                                               CLIPProj, TextDrivenDecoder, TextDecoder, MotionProj, MultiLabelClassifier)
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP
from tqdm import tqdm
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# custom functions #
import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()
import warnings
from diffusers import AutoencoderKL
import torch.nn.functional as F
from animatediff.utils.util import save_videos_grid
import json
from einops import rearrange, repeat
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score

def parse_arg():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="will load ckpt for model found in ../train_logs/model_name",
    )
    parser.add_argument(
        "--data_path", type=str, default=os.getcwd(),
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--root_dir", type=str, default='../cc2017_dataset',
    )
    parser.add_argument(
        "--weights_dir", type=str, default='/codes/NeuroClips/pretrained_weights',
    )
    parser.add_argument(
        "--exp", type=str, default='./saved_weights',
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--cache_dir", type=str, default='./pretrained_weights',
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
    )
    parser.add_argument(
        "--n_frames", type=int, default=6,
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=4096,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()
    return args



CLS_DICT = {
    0: 'animal',
    1: 'human',
    2: 'vehicle',
    3: 'building',
    4: 'clothing',
    5: 'weapon',
    6: 'plant',
    7: 'appliance',
    8: 'tool',
    9: 'container',
    10: 'body part',
    11: 'furniture',
    12: 'device',
    13: 'fabric',
    14: 'fruit',
    15: 'vegetable',
    16: 'insect',
    17: 'landscape feature',
    18: 'water body',
    19: 'organism',
    20: 'fish',
    21: 'reptile',
    22: 'mammal',
    23: 'accessory',
    24: 'sports equipment',
    25: 'food',
    26: 'drink',
    27: 'light source',
    28: 'weather phenomenon',
    29: 'jewelry',
    30: 'musical instrument',
    31: 'structure',
    32: 'flying vehicle',
    33: 'toy',
    34: 'kitchen item',
    35: 'writing tool',
    36: 'gardening tool',
    37: 'scientific equipment',
    38: 'furniture accessory',
    39: 'roadway',
    40: 'weaponry accessory',
    41: 'sports field',
    42: 'money',
    43: 'timekeeping device',
    44: 'decoration',
    45: 'art',
    46: 'stationery',
    47: 'kitchen appliance',
    48: 'rock/mineral',
    49: 'soil/substrate',
    50: 'climate/atmosphere component'
}





class CC2017_Dataset(torch.utils.data.Dataset):
    def __init__(self, voxel, image, text, text_embs, mask=None, cls_id=None, key_obj_cls=None, is_train=False, is_val=False):
        self.mask = mask
        self.key_obj_cls = key_obj_cls
        self.length = 1200
        self.is_train = is_train
        self.voxel = voxel
        self.image = image
        self.text = text
        self.cls_id = cls_id
        self.text_embs = text_embs


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        cls_label = torch.tensor(self.cls_id[idx]['category_id'])

        masks = self.mask[idx]
        masks[masks > 0] = 1
        cls = self.key_obj_cls[str(idx)]['category']

        sample = dict(pixel_values=self.image[idx], voxel=self.voxel[idx],
                      text=self.text_embs[idx], key_obj_masks=masks, key_obj_cls=cls,
                      cls_label=cls_label)
        return sample


def Decoding(model,clip_features):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    entry_length = 30
    temperature = 1
    tokens = None
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for i in range(entry_length):
        # print(location_token.shape)
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==49407:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _Tokenizer.decode(output_list)

        # output = tokenizer.decode(output_list, skip_special_tokens=True)


    except:
        output = 'None'
    return output


def prepare_dataset(args):



    voxel_test = torch.load(f'{args.root_dir}/subj0{args.subj}_test_fmri.pt', map_location='cpu')
    voxel_test = torch.mean(voxel_test, dim=1)
    print("Loaded all fmri test frames to cpu!", voxel_test.shape)
    test_images = torch.load(f'{args.root_dir}/GT_test_3fps.pt', map_location='cpu')
    test_text = torch.load(f'{args.root_dir}/GT_test_caption_qwen.pt', map_location='cpu')
    test_text_emb = torch.load(f'{args.root_dir}/GT_test_caption_qwen_emb.pt', map_location='cpu')

    print("Loaded all crucial test frames to cpu!", test_images.shape)

    key_objects_categories = json.load(open(f'{args.root_dir}/masks/key_objects_info_qwen_test.json'))
    print("Loaded all key_objects_categories to cpu!", len(key_objects_categories))
    key_objects_masks = torch.load(f'{args.root_dir}/masks/key_objects_masks_qwen_test.pt', map_location='cpu')
    print("Loaded all key_objects_masks to cpu!", key_objects_masks.shape)


    test_cls_id_json = json.load(open(f'{args.root_dir}/qwen_test_caption_tag_category_id.json'))
    print("Loaded all test class_labels_json to cpu!", len(test_cls_id_json))

    test_dataset = CC2017_Dataset(voxel_test, test_images, test_text, test_text_emb, mask=key_objects_masks, cls_id=test_cls_id_json, key_obj_cls=key_objects_categories, is_train=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    return test_dl, voxel_test





def prepare_brain_model(args):
    clip_seq_dim = 256
    clip_emb_dim = 1664
    seq_len = 1
    clip_txt_emb_dim = 1280

    clip_txt_embedder = FrozenOpenCLIPEmbedder2(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        layer="last",
        legacy=False,
        always_return_pooled=True,
        cache_dir="./pretrained_weights"

    )
    clip_txt_embedder.to(device)

    model = Neurons()
    model.ridge = RidgeRegression([voxel_test.shape[-1]], out_features=args.hidden_dim, seq_len=seq_len)
    model.clipproj = CLIPProj()

    model.backbone = BrainModel(h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=seq_len, n_blocks=args.n_blocks,
                              clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim)

    utils.count_params(model.backbone)
    utils.count_params(model)

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = args.n_frames
    dim_head = 52
    heads = clip_emb_dim // 52  # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
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

    model.text_seg_dec = TextDrivenDecoder(clip_emb_dim, clip_txt_emb_dim)
    model.text_dec = TextDecoder(clip_txt_emb_dim)
    model.motion_proj = MotionProj(n_frames=args.n_frames, clip_size=clip_emb_dim)
    model.classifier = MultiLabelClassifier(in_channel_img=clip_emb_dim, in_channel_text=clip_txt_emb_dim,
                                            seq_len=clip_seq_dim, class_num=51)

    model.to(device)

    utils.count_params(model.diffusion_prior)
    utils.count_params(model)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, cache_dir=args.cache_dir,
                                        subfolder="vae").to(device)

    print(f"\033[92m vae loaded \033[0m")

    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    utils.count_params(vae)

    print("---resuming from last.pth ckpt---")


    checkpoint = torch.load(os.path.join("EXP",  f"exp_{args.exp}", f"subj_{args.subj}", "checkpoints", f"brain_model_prior_last.pth"), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"\033[92m Pretrained brain_model loaded from {os.path.join('EXP',  f'exp_{args.exp}/subj_{args.subj}', 'checkpoints', f'brain_model_prior_last.pth')} \033[0m")
    del checkpoint
    return model, clip_txt_embedder, vae




def inference(args, model, clip_txt_embedder, vae, test_dl):
    batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),  # jpg doesnt get used, it's just a placeholder
             "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
             "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    # out = diffusion_engine.conditioner(batch)
    # vector_suffix = out["vector"].to(device)
    # print("vector_suffix", vector_suffix.shape)

    # get all reconstructions
    model.to(device)
    model.eval().requires_grad_(False)

    DiceLoss = utils.DiceLoss().cuda()

    dice_score_list = []
    acc_list_np = None
    cls_pred_list = []
    cls_label_list = []


    num_samples_per_image = 1
    assert num_samples_per_image == 1
    index = 0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(test_dl, desc='batches'):

            voxel = batch['voxel']
            video = batch['pixel_values'].to(device)
            masks = batch['key_obj_masks'].to(device)
            cls_label = batch['cls_label'].to(device)
            key_obj_cls = batch['key_obj_cls']

            voxel = voxel.unsqueeze(1).to(device)
            voxel = voxel.float()

            voxel_ridge = model.ridge(voxel, 0)  # 0th index of subj_list
            _, clip_vision_embeds = model.backbone(voxel_ridge)

            prior_out = model.diffusion_prior.p_sample_loop(clip_vision_embeds.shape,
                                                            text_cond=dict(text_embed=clip_vision_embeds),
                                                            cond_scale=1., timesteps=100)

            prior_out = prior_out.to(device)

            motion_embeds = model.motion_proj(prior_out)


            clip_text_embeds = model.clipproj(motion_embeds.mean(1))
            clip_text_embeds_norm = nn.functional.normalize(clip_text_embeds.flatten(1), dim=-1)


            cls_pred = model.classifier(motion_embeds.mean(1).mean(1))

            cls_pred = torch.sigmoid(cls_pred)
            top1_cls = torch.argmax(cls_pred, dim=1)
            # print(f"\033[92m {top1_cls} \033[0m")

            cls_pred = (cls_pred > 0.5)








            class_indices = [np.where(sample == 1)[0] for sample in cls_pred.cpu().numpy()]
            class_names = [[CLS_DICT[idx] for idx in indices] for indices in class_indices]
            best_class_names = [CLS_DICT[cls] for cls in top1_cls.cpu().numpy()]

            _, key_obj_text_embed = clip_txt_embedder(best_class_names)
            # _, key_obj_text_embed = clip_txt_embedder(key_obj_cls)

            seg_masks = model.text_seg_dec(rearrange(motion_embeds, "b f n c -> (b f) n c"),
                                           key_obj_text_embed,
                                           time=args.batch_size * args.n_frames,
                                           is_seg=True)


            vae_embeds = model.text_seg_dec(rearrange(motion_embeds, "b f n c -> (b f) n c"),
                                            clip_text_embeds,
                                            time=args.batch_size * args.n_frames,
                                            is_seg=False)



            seg_masks = torch.sigmoid(seg_masks)
            seg_masks = (seg_masks > 0.5).float()
            seg_masks_cond = (seg_masks + 1) / 2

            seg_masks_cond = F.interpolate(seg_masks_cond, (28, 28), mode="nearest")
            seg_masks_save = F.interpolate(seg_masks, (224, 224), mode="nearest")
            vae_embeds = F.interpolate(vae_embeds, (28, 28), mode="nearest")
            masks_eval = F.interpolate(masks, (64, 64), mode="nearest")
            gt_masks_save = F.interpolate(masks, (224, 224), mode="nearest")



            blurry_recon_images_org = (vae.decode(vae_embeds / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            blurry_recon_images = (vae.decode(vae_embeds * seg_masks_cond / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            # print(f"\033[92m {blurry_recon_images.shape} \033[0m")

            blurry_recon_images = rearrange(blurry_recon_images, "(b f) c h w -> b f c h w", f= args.n_frames)
            blurry_recon_images_org = rearrange(blurry_recon_images_org, "(b f) c h w -> b f c h w", f= args.n_frames)
            seg_masks_video = rearrange(seg_masks, "(b f) c h w -> b f c h w", f= args.n_frames)
            seg_masks_video_save = rearrange(seg_masks_save, "(b f) c h w -> b f c h w", f= args.n_frames)
            seg_masks_loss = rearrange(seg_masks, "(b f) c h w -> b f c h w", f= args.n_frames)


            # print(f"\033[92m seg_masks_video.shape {seg_masks_video.shape} \033[0m")
            # Feed diffusion prior outputs through unCLIP
            for i in range(len(voxel)):
                print(index)

                cls_pred_list.append(cls_pred[i])
                cls_label_list.append(cls_label[i])

                class_pred, class_label = torch.stack(cls_pred_list, dim=0), torch.stack(cls_label_list, dim=0)
                # print(f"\033[92m {class_pred.shape} \033[0m")
                # print(f"\033[92m {class_label.shape} \033[0m")
                class_accuracy = {}

                # 遍历每个类别
                for class_idx in range(class_label.shape[1]):
                    # 预测值和真实值
                    pred_class = class_pred[:, class_idx].cpu().numpy()
                    true_class = class_label[:, class_idx].cpu().numpy()

                    # 计算预测正确的次数
                    correct = np.sum((pred_class == 1) & (true_class == 1))

                    # 计算真实标签为 1 的次数
                    total_true = np.sum(true_class == 1)

                    # 计算准确率（避免除以 0）
                    if total_true > 0:
                        accuracy = correct / total_true
                    else:
                        accuracy = 0.0  # 如果没有真实标签为 1 的样本，准确率为 0

                    # 将准确率保存到字典中
                    class_accuracy[class_idx] = accuracy



                print(f"\033[93m class_names {class_names[i]} \033[0m")
                print(f"\033[93m best_class_names {best_class_names[i]} \033[0m")
                print(f"\033[93m ACC: {len(class_pred), np.array(class_accuracy)} \033[0m")




                dice_score = 1 - DiceLoss(seg_masks_loss[i], masks_eval[i].unsqueeze(1))

                dice_score_list.append(dice_score.cpu().detach().numpy())

                print(f"\033[92m Dice: {np.mean(dice_score_list), len(dice_score_list)} \033[0m")


                video_save = rearrange(blurry_recon_images, "b f c h w -> b c f h w")[i].cpu()
                video_save_org = rearrange(blurry_recon_images_org, "b f c h w -> b c f h w")[i].cpu()
                cur_seg_masks_video_save = repeat(seg_masks_video_save, "b f c h w -> b f (r c) h w", r=3)
                cur_seg_masks_video_save = rearrange(cur_seg_masks_video_save, "b f c h w -> b c f h w")[i].cpu()

                cur_get_masks_video_save = repeat(gt_masks_save.unsqueeze(2), "b f c h w -> b f (r c) h w", r=3)
                cur_get_masks_video_save = rearrange(cur_get_masks_video_save, "b f c h w -> b c f h w")[i].cpu()\

                save_videos_grid(
                    torch.cat([video.permute(0, 2, 1, 3, 4)[i].unsqueeze(0).cpu(), video_save_org.unsqueeze(0), video_save.unsqueeze(0),
                               cur_get_masks_video_save.unsqueeze(0), cur_seg_masks_video_save.unsqueeze(0)]),
                    f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated_video_test/video_{index}.gif")

                # print(f"\033[92m {pred_text_norm[i].shape} \033[0m")
                generated_text = Decoding(model.text_dec, clip_text_embeds_norm[i])
                generated_text = generated_text.replace('<|startoftext|>', '').replace('<|endoftext|>', '')

                print(f"{generated_text}")



                index += 1




if __name__ == "__main__":
    args = parse_arg()

    # seed all random functions
    utils.seed_everything(args.seed)

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



    model_name = f'video_subj0{args.subj}'

    test_dl, voxel_test = prepare_dataset(args)
    model, clip_txt_embedder, vae = prepare_brain_model(args)


    os.makedirs(f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated_test", exist_ok=True)
    os.makedirs(f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated_img_test", exist_ok=True)


    inference(args, model, clip_txt_embedder,  vae, test_dl)

    if not utils.is_interactive():
        sys.exit(0)