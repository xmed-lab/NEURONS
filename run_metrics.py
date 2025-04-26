import os, sys
import numpy as np
from eval_metrics import (clip_score_only,
                          compute_clip_pcc,
                          clip_score_frame,
                          ssim_score_only,
                          psnr_score_only,
                          img_classify_metric,
                          video_classify_metric,
                          remove_overlap)
import imageio.v3 as iio
import torch
import argparse
from tqdm import tqdm

import sys

sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP


def main(
        data_path,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


 
    gt_list = []
    pred_list = []
    # for i  in range(1200):
    #     gif = iio.imread(os.path.join(data_path, f'test{i+1}.gif'), index=None)
    #     gt, pred = np.split(gif, 2, axis=2)
    #     gt_list.append(gt)
    #     pred_list.append(pred)

    for filename in os.listdir(data_path):
        if not filename.endswith('.gif'):
            continue
        gif = iio.imread(os.path.join(data_path, filename), index=None)
        gt, pred = np.split(gif, 2, axis=2)
        gt_list.append(gt)
        pred_list.append(pred)


    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)

    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    # image classification scores
    num_trials = 100
    top_k = 1
    # video classification scores

    clip_pcc_mean, clip_pcc_std = compute_clip_pcc(pred_list)
    clip_pcc_mean_new, clip_pcc_std_new = clip_score_frame(pred_list)

    acc_list_2way_v, std_list_2way_v = video_classify_metric(
                                        pred_list,
                                        gt_list,
                                        n_way = 2,
                                        top_k=top_k,
                                        num_trials=num_trials,
                                        num_frames=gt_list.shape[1],
                                        return_std=True,
                                        device=device
                                        )

    acc_list_50way_v, std_list_50way_v = video_classify_metric(
                                        pred_list,
                                        gt_list,
                                        n_way = 50,
                                        top_k=top_k,
                                        num_trials=num_trials,
                                        num_frames=gt_list.shape[1],
                                        return_std=True,
                                        device=device
                                        )


    print(f"\033[92m ======== Video-based ======== \033[0m")
    print(f"\033[92m \t-------- Sematic-level -------- \033[0m")
    print(f'\033[92m \t\t 2-way: {np.mean(acc_list_2way_v)} ± {np.mean(std_list_2way_v)} \033[0m')
    print(f'\033[92m \t\t50-way: {np.mean(acc_list_50way_v)} ± {np.mean(std_list_50way_v)} \033[0m')

    print(f"\033[92m \t-------- ST-level -------- \033[0m")
    print(f'\033[92m \t\t CLIP-pcc: {clip_pcc_mean} ± {clip_pcc_std} \033[0m')
    print(f'\033[92m \t\t CLIP-pcc new: {clip_pcc_mean_new} ± {clip_pcc_std_new} \033[0m')



    ssim_scores_list = []
    ssim_scores_std_list = []
    psnr_scores_list = []
    psnr_scores_std_list = []
    frame_acc_list_2way = []
    frame_acc_list_50way = []
    frame_acc_std_list_2way = []
    frame_acc_std_list_50way = []

    for i in tqdm(range(pred_list.shape[1])):

        # ssim scores
        ssim_scores, ssim_std = ssim_score_only(pred_list[:, i], gt_list[:, i])
        psnr_scores, psnr_std = psnr_score_only(pred_list[:, i], gt_list[:, i])
        # print(f'ssim score: {ssim_scores}, std: {ssim_std}')

        ssim_scores_list.append(ssim_scores)
        ssim_scores_std_list.append(ssim_std)

        psnr_scores_list.append(psnr_scores)
        psnr_scores_std_list.append(psnr_std)

        acc_list_2way, std_list_2way = img_classify_metric(
                                            pred_list[:, i],
                                            gt_list[:, i],
                                            n_way = 2,
                                            top_k=top_k,
                                            num_trials=num_trials,
                                            return_std=True,
                                            device=device)

        acc_list_50way, std_list_50way = img_classify_metric(
            pred_list[:, i],
            gt_list[:, i],
            n_way=50,
            top_k=top_k,
            num_trials=num_trials,
            return_std=True,
            device=device)

        frame_acc_list_2way.append(np.mean(acc_list_2way))
        frame_acc_std_list_2way.append(np.mean(std_list_2way))
        frame_acc_list_50way.append(np.mean(acc_list_50way))
        frame_acc_std_list_50way.append(np.mean(std_list_50way))



    print(f"\033[92m ======== Frame-based ======== \033[0m")
    print(f"\033[92m \t-------- Sematic-level -------- \033[0m")
    print(f'\033[92m \t\t 2-way: {np.mean(frame_acc_list_2way)} ± {np.mean(frame_acc_std_list_2way)} \033[0m')
    print(f'\033[92m \t\t50-way: {np.mean(frame_acc_list_50way)} ± {np.mean(frame_acc_std_list_50way)} \033[0m')
    print(f"\033[92m \t-------- Pixel-level -------- \033[0m")
    print(f'\033[92m \t\t SSIM: {np.mean(ssim_scores_list)} ± {np.mean(ssim_scores_std_list)} \033[0m')
    print(f'\033[92m \t\t PSNR: {np.mean(psnr_scores_list)} ± {np.mean(psnr_scores_std_list)} \033[0m')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="/codes/NeuroClips/Animatediff/StableDiffusion", )
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument(
        "--root_dir", type=str, default='/data/cc2017_dataset',
    )
    parser.add_argument(
        "--exp", type=str, default='', required=True
    )
    parser.add_argument(
        "--mode", type=str, default=''
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3],
        help="Validate on which subject?",
    )
    args = parser.parse_args()



    if args.mode == "self":
        data_path = f'./EXP/exp_{args.exp}/subj_{args.subj}/gen_videos_motion_self'
    elif args.mode == "motion":
        data_path = f'./EXP/exp_{args.exp}/subj_{args.subj}/gen_videos_motion'
    elif args.mode == "cap":
        data_path = f'./EXP/exp_{args.exp}/subj_{args.subj}/gen_videos'
    else:
        data_path = f'./EXP/exp_{args.exp}/subj_{args.subj}/gen_videos_{args.mode}'


    print(f"\033[92m Evaluating results from: {data_path} \033[0m")

    main(data_path=data_path)