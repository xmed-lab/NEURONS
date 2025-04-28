#!/bin/sh
eval "$(conda shell.bash hook)"
export MASTER_PORT=$((RANDOM % 64512 + 1024))
export CUDA_VISIBLE_DEVICES=$1
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))
export PYTHONPATH=$(pwd)/:$PYTHONPATH


echo $num_gpus

EXP_ROOT_DIR="EXP"

exp=$2
stage=$3
mode=$4
subj=$5


if [ ! -d "./${EXP_ROOT_DIR}" ]; then
  mkdir ./${EXP_ROOT_DIR}
fi

if [ ! -d "./${EXP_ROOT_DIR}/exp_${exp}" ]; then
  mkdir ./${EXP_ROOT_DIR}/exp_${exp}
fi

if [ ! -d "./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj" ]; then
  mkdir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj
fi

timestamp=$(date +"%H:%M_%Y-%m-%d")


conda activate neurons_train


if [[ "$stage" == *"1"* ]];
  then
    echo $stage
      python -u \
            train_neurons.py \
            --subj $subj \
            --batch_size 120 \
            --num_epochs 30 \
            --mixup_pct 1.0 \
            --max_lr 5e-5 \
            --exp_dir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj \
            | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/backbone_log_$timestamp.txt
fi



if [[ "$stage" == *"2"* ]];
  then
    echo $stage
      python -u \
              train_neurons.py \
              --subj $subj \
              --batch_size 10 \
              --num_epochs 50 \
              --mixup_pct 0.0 \
              --max_lr 5e-5 \
              --exp_dir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj \
              --neurons_decoupler \
              | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/prior_log_$timestamp.txt
fi


if [[ "$stage" == *"3"* ]];
  then
    if [[ "$mode" == *"enhance"* ]]; then
      python -u recon_keyframe_neurons_enhance.py --subj $subj --exp $exp | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/recon_enhance_log_$timestamp.txt
    else
      python -u recon_keyframe_neurons.py --subj $subj --exp $exp | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/recon_log_$timestamp.txt
    fi
fi

if [[ "$stage" == *"4"* ]];
  then
    if [[ "$mode" == *"enhance"* ]]; then
      python -u caption_keyframe_enhance.py --subj $subj --exp $exp | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/cap_log_$timestamp.txt
    else
      python -u caption_keyframe.py --subj $subj --exp $exp | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/cap_log_$timestamp.txt
    fi
fi

if [[ "$stage" == *"5"* ]];
  then
    conda activate neurons_test
    if [[ "$mode" == *"enhance"* ]]; then
      accelerate launch --main_process_port $MASTER_PORT \
          scripts/neuroclips_video_enhance.py --exp ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj --mode $mode --subj $subj  | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/video_enhance_log_$timestamp.txt
    else
      accelerate launch --main_process_port $MASTER_PORT \
          scripts/neuroclips_video.py --exp ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj --mode $mode --subj $subj  | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/video_log_$timestamp.txt
    fi
fi

if [[ "$stage" == *"6"* ]];
  then
    python run_metrics.py --exp $exp  --mode $mode --subj $subj | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/eval_res_$mode.txt
fi


if [[ "$stage" == *"e"* ]];
  then
    python -u gen_decoupled_outputs.py --subj $subj --exp $exp | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/recon_log_$timestamp.txt
fi
