# <span style="font-variant: small-caps;">Neurons</span>: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.11167-brown?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2503.11167)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/McGregorW/NEURONS)

🌟 **If you find our project useful, please consider giving us a star!** ⭐

</div>


## 📌 Overview


<div align="center">

![model](assets/framework.png)
*Architecture of the <span style="font-variant: small-caps;">Neurons</span> framework*

</div>


<span style="font-variant: small-caps;">Neurons</span> is a novel framework that emulates the human visual cortex to achieve high-fidelity and interpretable fMRI-to-video reconstruction. Our biologically-inspired approach significantly advances the state-of-the-art in brain decoding and visual reconstruction.



## 📣 Latest Updates



<div style="display: flex; align-items: center;">
    <img src="https://img.shields.io/badge/2025%2F04-yellow?style=flat-square" alt="Static Badge" style="margin-right: 8px;">
    <span>Code released</span>
</div>

<br>

<div style="display: flex; align-items: center;">
    <img src="https://img.shields.io/badge/2025%2F03-yellow?style=flat-square" alt="Static Badge" style="margin-right: 8px;">
    <span>Project launched with paper available on arXiv</span>
</div>

## 🛠️ Installation & Setup


### 🖥️ Environment Setup

We recommend using separate environments for training and testing:

```bash
# Training environment
conda create -n train python==3.10
conda activate train
pip install -r requirements.txt

# Testing environment (to avoid package conflicts)
conda create -n test --clone train
conda activate test
pip install diffusers==0.11.1
```


### 📊 Data Preparation


1. Download the pre-processed dataset from [NeuroClips](https://github.com/gongzix/NeuroClips):

```bash
python download_dataset.py
tar -xzvf ./cc2017_dataset/masks/mask_cls_train_qwen_video.tar.gz -C ./cc2017_dataset/masks/
tar -xzvf ./cc2017_dataset/masks/mask_cls_test_qwen_video.tar.gz -C ./cc2017_dataset/masks/
```

2. Run task construction scripts:

```bash
# Rule-based Key Object Discovery
python tasks_construction/find_key_obj.py

# Generate CLIP embeddings
python -m tasks_construction.gen_GT_clip_embeds
```


### ⚙️ Pretrained Weights Preparation

```shell
mkdir pretrained_weights
cd pretrained_weights
wget -O unclip6_epoch0_step110000.ckpt -c https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/unclip6_epoch0_step110000.ckpt\?download\=true
wget -O last.pth -c https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj01_pretrained_40sess_24bs/last.pth\?download\=true
wget -O convnext_xlarge_alpha0.75_fullckpt.ckpt -c https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/convnext_xlarge_alpha0.75_fullckpt.pth\?download\=true
wget -O sd_image_var_autoenc.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/sd_image_var_autoenc.pth?download=true
```


## 🚀 Quick Start
This codebase allows train, test, and evaluate using one single bash file.

```
bash train_neurons.sh 0 neurons 123456 enhance 1
```


------
Parameters:

`$1`: use which gpu to train

`$2`: train file postfix, e.g, `train_neurons`

- `1`: train brain model
- `2`: train decoupler
- `3`: recon decoupled outputs, prepare for video reconstruction
- `4`: (Optional) caption the keyframes with BLIP-2 instead of using the outputs of GPT-2 in Neurons
- `5`: video reconstruction
- `6`: evaluation with all metrics

`$3`: run which stage: `123456` for the whole process, `3456` for test & eval only

`$4`: inference mode: `['enhance', 'motion']`

`$5`: train which subject: `[0,1,2]`

----
Note that for convenience of debugging, `use_wandb` is set to `False` be default. 

If you would like to use wandb, first run `wandb login` and set the `use_wandb` to `True` in `train_neurons.py`.




## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{wang2025neurons,
  title={NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction},
  author={Wang, Haonan and Zhang, Qixiang and Wang, Lehan and Huang, Xuanqi and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2503.11167},
  year={2025}
}
```