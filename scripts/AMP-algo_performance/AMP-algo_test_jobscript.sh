#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=AMP-algo_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dtirinna@sissa.it
#SBATCH --output=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/scripts/AMP-algo_performance/out/%x.%j.out
#SBATCH --error=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/scripts/AMP-algo_performance/out/%x.%j.err

source $HOME/virtualenvs/dl/bin/activate

cd $HOME/HUPLACLIP-NNs/scripts/AMP-algo_performance

torchrun --standalone --nproc_per_node=1 AMP-algo_script.py