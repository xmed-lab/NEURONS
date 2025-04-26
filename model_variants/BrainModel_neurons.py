import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import clip
import random
from tqdm import tqdm
import math
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from .video_decoder import DecoderVideo

class MultiLabelClassifier(nn.Module):
    def __init__(self, in_channel_img, in_channel_text, seq_len, class_num):
        super().__init__()
        # self.vision_proj_seq = nn.Linear(seq_len, 1)
        self.vision_proj_channel = nn.Linear(in_channel_img, in_channel_text)
        self.classifier = nn.Linear(in_channel_text, class_num)


    def forward(self, x_i):
        x_i = self.vision_proj_channel(x_i)
        x = self.classifier(x_i)
        return x



class TextDrivenDecoder(nn.Module):
    def __init__(self, clip_vision_emb_dim, clip_txt_emb_dim, attention_dropout_rate=0.1):
        super().__init__()

        self.q = nn.Linear(clip_vision_emb_dim, clip_txt_emb_dim, bias=False)
        self.k = nn.Linear(clip_txt_emb_dim, clip_txt_emb_dim, bias=False)
        self.v = nn.Linear(clip_txt_emb_dim, clip_txt_emb_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.out = nn.Linear(clip_txt_emb_dim, clip_txt_emb_dim, bias=False)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.dropout = nn.Dropout(.3)
        self.norm = nn.GroupNorm(1, 64)



        self.maps_projector = nn.Sequential(
            nn.Conv2d(clip_txt_emb_dim, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 128, 1, bias=False),
            nn.GroupNorm(1, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1, bias=True),
        )

        self.video_decoder = DecoderVideo(
            in_channels=64,
            up_block_types=["AttnUpDecoderBlock2D", "AttnUpDecoderBlock2D", "AttnUpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )

        self.recon_head = nn.Conv2d(32, 4, 3, padding=1)
        self.seg_head = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, vision_feat, text_feat=None, time=1, is_seg=True, return_all=False):

        if text_feat is not None:
            q = self.q(vision_feat)
            k = self.k(text_feat)
            v = self.v(text_feat)
            scale = vision_feat.shape[-1] ** -0.5
            cross_attn = torch.matmul(q, k.transpose(-1, -2))
            cross_attn = self.attn_dropout(self.softmax(cross_attn * scale))
            cross_attn = torch.matmul(cross_attn, v)
            out = self.out(cross_attn)
            vision_feat = self.proj_dropout(out)
        else:
            q = self.q(vision_feat)
            out = self.out(q)
            vision_feat = self.proj_dropout(out)



        B, N, C = vision_feat.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        vision_feat = rearrange(vision_feat, "b (h w) c -> b c h w", h=H, w=W)



        x = self.maps_projector(vision_feat)


        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1, H, W).contiguous()
        x = self.norm(x)
        # x_aux = self.maps_projector(x).flatten(2).permute(0, 2, 1)
        # x_aux = x_aux.view(len(x_aux), 49, 512)
        x = self.video_decoder(x, time=time)


        if is_seg:
            x_seg = self.seg_head(x)
            return x_seg
        elif return_all:
            x_seg = self.seg_head(x)
            x_recon = self.recon_head(x)
            return x_seg, x_recon
        else:
            x_recon = self.recon_head(x)
            return x_recon



class MotionProj(nn.Module):
    def __init__(self, n_frames=6, clip_size=768):
        super().__init__()
        self.n_frames = n_frames
        self.clip_size = clip_size
        self.motion_proj = nn.Linear(clip_size, clip_size * n_frames, bias=True)


    def forward(self, x):
        # x: [b, 256, 1664]
        motion_embeds = self.motion_proj(x)
        motion_embeds = rearrange(motion_embeds, 'b n (c f) -> b c f n', f=self.n_frames)

        B, C, F, N = motion_embeds.shape

        # print(f"\033[91m motion_embeds {motion_embeds.shape} \033[0m")

        motion_embeds = motion_embeds.view(B, C, F, int(math.sqrt(N)), int(math.sqrt(N)))

        # print(f"\033[92m {motion_embeds.shape} \033[0m")

        motion_embeds = rearrange(motion_embeds, 'b c f h w -> b f (h w) c')

        # print(f"\033[93m {motion_embeds.shape} \033[0m")



        return motion_embeds



class text_MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.GELU):
        super(text_MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class TextDecoder(nn.Module):

    def __init__(self, prefix_size: int = 1280):
        super(TextDecoder, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        # with open('./decoder_config.pkl', 'rb') as f:
        #     config = pickle.load(f)
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = text_MLP((prefix_size, self.embedding_size))

    def forward(self, clip_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        # print(f"\033[92m == embedding_text {embedding_text.shape} \033[0m")
        embedding_clip = self.clip_project(clip_features)
        # print(f"\033[92m == embedding_clip {embedding_clip.shape} \033[0m")

        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        # print(f"\033[92m == embedding_clip2 {embedding_clip.shape} \033[0m")
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        # print(f"\033[92m out {out.shape} \033[0m")
        return out





class CLIPProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1664, 1280))

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = x @ self.proj
        # print(f"\033[92m x @ self.proj {x.shape} \033[0m")
        return x


class Neurons(nn.Module):
    def __init__(self):
        super(Neurons, self).__init__()

    def forward(self, x):
        return x


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.seq_len = seq_len
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(self.seq_len)], dim=1)
        return out



class BrainModel(nn.Module):
    def __init__(self, h=4096, in_dim=13447, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768,
                 blurry_recon=True, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])

        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True)
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)

    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )


    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )

    def forward(self, x):
        # print(f"\033[92m ===== backbone forward ===== \033[0m")
        # x: [60, 1, 4096]

        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)

            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)

        x = x.reshape(x.size(0), -1)
        # x: [60, 4096]

        voxels_embed = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        # backbone = self.bn1(backbone)
        # backbone: [60, 256, 1664]

        clip_vision_embed = self.clip_proj(voxels_embed)
        return voxels_embed, clip_vision_embed



# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, \
    prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward


class BrainDiffusionPrior(DiffusionPrior):
    """
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """

    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.,
                 generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, text_cond=text_cond,
                                                                          self_cond=self_cond,
                                                                          clip_denoised=clip_denoised,
                                                                          cond_scale=cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed  # / self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond,
                                                 cond_scale=cond_scale,
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target)  # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
            self,
            text=None,
            image=None,
            voxel=None,
            text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
            image_embed=None,
            text_encodings=None,  # as well as CLIP text encodings
            *args,
            **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(
            voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(
            text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)

        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred


class PriorNetwork(nn.Module):
    def __init__(
            self,
            dim,
            num_timesteps=None,
            num_time_embeds=1,
            # num_image_embeds = 1,
            # num_brain_embeds = 1,
            num_tokens=257,
            causal=True,
            learned_query_mode='none',
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob=1., image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            brain_embed=None,
            text_embed=None,
            brain_cond_drop_prob=0.,
            text_cond_drop_prob=None,
            image_cond_drop_prob=0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob


        # print(f"\033[92m ==== image_embed {image_embed.shape} \033[0m")
        # print(f"\033[92m ==== brain_embed {brain_embed.shape} \033[0m")

        # image_embed = image_embed.view(len(image_embed),-1,16*16)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,16*16)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)

        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds

        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # print(f"\033[92m ==== after brain_embed {brain_embed.shape} \033[0m")


        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention
        # (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b=batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)

        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim=-2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed


class FlaggedCausalTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            norm_in=False,
            norm_out=True,
            attn_dropout=0.,
            ff_dropout=0.,
            final_proj=True,
            normformer=False,
            rotary_emb=True,
            causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=causal, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                          rotary_emb=rotary_emb),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))

        self.norm = LayerNorm(dim,
                              stable=True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

