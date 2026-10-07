#!/bin/bash
set -e
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate /public/home/yingxy2024/.conda/envs/minwm
export PATH="/public/home/yingxy2024/.conda/envs/minwm/bin:$PATH"
hash -r
cd /public/home/yingxy2024/minWM
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"
exec bash Wan21/scripts/training/run_stage1_ar_camera.sh
