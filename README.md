# <span style="font-variant: small-caps;">Neurons</span>: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction


## 🛠️ Method
![model](assets/framework.png)


## Installation
```
conda create -n train python==3.10
conda activate train
pip install -r requirements.txt
```

There exists package conflicts of training Neurons and testing with Animatediff pipeline. Therefore, we need to clone another virtual environment:

```
conda create -n test --clone train
pip install diffusers==0.11.0
```

## Run
This codebase allows train, test, and evaluate using one single bash file.

```
bash train_neurons.sh 0 neurons 123456 enhance 1
```
Parameters:

`$1`: use which gpu to train

`$2`: train file postfix, e.g, `train_neurons`

- `1`: train brain model
- `2`: train decoupler
- `3`: recon decoupled outputs, prepare for video reconstruction
- `4`: (Optional) caption the keyframes with BLIP-2 instead of using the outputs of GPT-2 in Neurons
- `5`: video reconstruction
- `6`: evaluation with all metrics

`$3`: run which stage

`$4`: inference mode `['enhance', 'motion']`

`$5`: train which subject: `[0,1,2]`
