#!/bin/bash


#SBATCH --nodes 1
#SBATCH --job-name=trainProtoPNetFast
#SBATCH --cpus-per-task 4


python main-fast.py
