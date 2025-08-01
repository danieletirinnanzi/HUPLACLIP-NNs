#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=cnn_exp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dtirinna@sissa.it
#SBATCH --output=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/out/%x.%j.out
#SBATCH --error=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/out/%x.%j.err

source $HOME/virtualenvs/dl/bin/activate

cd $HOME/HUPLACLIP-NNs/
# cd $SLURM_SUBMIT_DIR

torchrun --standalone --nproc_per_node=4 main.py
