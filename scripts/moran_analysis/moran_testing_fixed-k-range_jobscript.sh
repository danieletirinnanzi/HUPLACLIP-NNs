#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=Sis26_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=moran_testing_fixed-k-range
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dtirinna@sissa.it
#SBATCH --output=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/scripts/moran_analysis/out/%x.%j.out
#SBATCH --error=/leonardo/home/userexternal/dtirinna/HUPLACLIP-NNs/scripts/moran_analysis/out/%x.%j.err

source $HOME/virtualenvs/dl/bin/activate

cd $HOME/HUPLACLIP-NNs/scripts/moran_analysis/
# cd $SLURM_SUBMIT_DIR

python nns_moran_testing_fixed-k-range.py