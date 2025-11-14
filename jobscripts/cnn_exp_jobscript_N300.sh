#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=cnn_exp_N300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dtirinna@sissa.it
#SBATCH --output=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/out/N300/%x.%j.out
#SBATCH --error=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/out/N300/%x.%j.err

source $HOME/virtualenvs/dl/bin/activate

cd $HOME/HUPLACLIP-NNs/
# cd $SLURM_SUBMIT_DIR

# --------- HOW TO RUN THIS SCRIPT (uncomment the line corresponding to the experiment to be run) ---------
# NOTE: in both cases, the configuration file in "test_models.py" (line 16) must match the one used here
# - STANDARD CASE:
# torchrun --standalone --nproc_per_node=4 main.py --config docs/cnn_exp_config_N300.yml
# - RESUME CASE (for continuing a previous experiment):
torchrun --standalone --nproc_per_node=4 main.py --resume --exp_name cnn_exp_N300_2025-11-03_09-57-46
# - TESTING TRAINED MODEL:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # to avoid CUDA out of memory error
# torchrun --standalone --nproc_per_node=4 -m tests.test_loop.testloop_test_DDP
