import torch
import numpy as np
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP
import json
from tqdm import tqdm



def gen_captions_embeds(mode, clip_embedder):
    qwen_captions = json.load(open(f'./tasks_construction/qwen_{mode}_caption_tag.json'))
    all_predcaptions = []
    all_predcaptions_emb = []
    for dict in tqdm(qwen_captions):
        caption = dict['caption']
        # print(f"\033[92m {caption} \033[0m")
        all_predcaptions = np.hstack((all_predcaptions, caption))

        _, embedding = clip_embedder(caption)
        # print(f"\033[91m {embedding.shape} \033[0m")
        embedding = embedding.cpu()
        all_predcaptions_emb.append(embedding[0])


    print(f"\033[92m all_predcaptions {all_predcaptions.shape} \033[0m")
    torch.save(all_predcaptions, f'{root_dir}/GT_{mode}_caption_qwen.pt')

    all_predcaptions_emb = torch.stack(all_predcaptions_emb)
    print(f"\033[92m all_predcaptions_emb {all_predcaptions_emb.shape} \033[0m")
    torch.save(all_predcaptions_emb, f'{root_dir}/GT_{mode}_caption_qwen_emb.pt')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = './cc2017_dataset/qwen_annotation'

    clip_embedder = FrozenOpenCLIPEmbedder2(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        layer="last",
        legacy=False,
        always_return_pooled=True,
        cache_dir="./pretrained_weights"
    )
    clip_embedder.to(device)


    gen_captions_embeds("train", clip_embedder)
    gen_captions_embeds("test", clip_embedder)
