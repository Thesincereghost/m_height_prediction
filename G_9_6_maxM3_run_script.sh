#!/bin/bash
#SBATCH --job-name=G_9_6_maxM3   # Job name
#SBATCH --mail-type=END,FAIL         # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=vaishnav.g@tamu.edu   # Replace with your email address
#SBATCH --ntasks=36                   # Run on a single CPU
#SBATCH --ntasks-per-node=36 	## Request exactly (or max) of 36 tasks per node
#SBATCH --mem=36G                 # Request 2560MB (2.5GB) per node
#SBATCH --time=32:00:00              # Time limit hh:mm:ss
#SBATCH --output=%x.log              # Standard output and error log
module load Anaconda3/2024.02-1
conda activate deeplearning_636

python compute_m_heights.py generator_matrices/G_9_6_maxM3.pkl.gz 920 46
