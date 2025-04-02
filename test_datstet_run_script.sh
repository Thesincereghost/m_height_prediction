#!/bin/bash
#SBATCH --job-name=test_dataset   # Job name
#SBATCH --mail-type=END,FAIL         # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=vaishnav.g@tamu.edu   # Replace with your email address
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --mem=2560M                  # Request 2560MB (2.5GB) per node
#SBATCH --time=12:00:00              # Time limit hh:mm:ss
#SBATCH --output=%x.log              # Standard output and error log
module purge
module laod Anaconda3/2024.02-1
conda init
bash
conda activate deeplearning_636

python compute_m_heights.py generator_matrices/test_dataset/test_dataset_n9_k4_m5.pkl.gz 2 2
